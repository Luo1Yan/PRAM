"""Extract ImageBind features from model-ready sensor tensors.

Run python scripts/extract_imagebind.py --manifest relation.json --output pack.npz
after installing the official ImageBind package.

The JSON manifest contains RelationPack metadata and split fields plus a streams
object with a, p, b and optional p_pb entries. For example, a depth stream is
{"modality":"depth","inputs":"depth.npy","layout":"spatial","grid_size":14}.
Each input is a row-major NPY file or an ordered list of per-sample NPY files.
Use official checkpoint preprocessing. Split, label and coordinate fields accept
inline arrays or paths to NPY files relative to the manifest. Query dictionaries
map set names to arrays or paths. All multi-part streams must use the same grid.

Image and video samples have shapes [C,H,W] and [C,T,H,W]. Audio uses [1,F,T],
depth and thermal use [1,H,W], IMU uses [6,T], and text uses token IDs [L].
A leading clip axis is optional. Layout selects spatial, temporal or none.
A one-based block selects intermediate local tokens. Global vectors always pass
through the complete frozen tower. Set encoder.checkpoint to a local state dict
to select explicit weights. Otherwise the official pretrained checkpoint is used.

For three spatial streams, an optional registration object accepts max_shift,
max_frames, seed and pool_size. Its offsets are estimated on fitting edges only.
Upstream model and preprocessing: https://github.com/facebookresearch/ImageBind
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import RelationPack, save_relation_pack, split_disjoint_halves
from pram import align_spatial_grids, estimate_spatial_offset


def spatial_pool(grid, size):
    if type(size) is not int or size < 1:
        raise ValueError("grid_size must be a positive integer")
    pooled = F.adaptive_avg_pool2d(grid.permute(0, 3, 1, 2), (size, size))
    return pooled.permute(0, 2, 3, 1).contiguous()


def temporal_pool(sequence, segments):
    if type(segments) is not int or not 1 <= segments <= sequence.size(1):
        raise ValueError("Each temporal segment requires at least one observation")
    return torch.stack([chunk.mean(1) for chunk in sequence.tensor_split(segments, dim=1)], 1)


class ImageBindEncoder:
    def __init__(self, device, options, root):
        from imagebind.models.imagebind_model import imagebind_huge

        checkpoint = options.get("checkpoint")
        self.model = imagebind_huge(pretrained=checkpoint is None)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(root / checkpoint, map_location="cpu", weights_only=True))
        self.model.to(device).eval().requires_grad_(False)

    def __call__(self, spec, value):
        modality = spec["modality"].lower()
        key = {"image": "vision", "video": "vision", "rgb": "vision"}.get(modality, modality)
        if key not in self.model.modality_trunks:
            raise ValueError(f"Unknown ImageBind modality {modality}")
        rank = {"video": 5, "imu": 3, "text": 2}.get(modality, 4)
        n = len(value)
        if value.ndim == rank + 1:
            value = value.flatten(0, 1)
        elif value.ndim != rank:
            raise ValueError(f"Unexpected input shape for {modality}: {tuple(value.shape)}")
        value = value.long() if key == "text" else value.float()
        prep = self.model.modality_preprocessors[key]
        inputs = prep(**{key: value})
        trunk = self.model.modality_trunks[key]
        layout = spec.get("layout", {
            "text": "none", "imu": "temporal", "audio": "temporal", "video": "temporal",
        }.get(modality, "spatial"))
        block = spec.get("block")
        capture = []
        handle = None
        if block is not None and layout != "none":
            if type(block) is not int or not 1 <= block <= len(trunk.blocks):
                raise ValueError(f"Invalid ImageBind block {block}")
            handle = trunk.blocks[block - 1].register_forward_hook(
                lambda module, args, output: capture.append(output.transpose(0, 1))
            )
        try:
            tokens = trunk(**inputs["trunk"])
        finally:
            if handle is not None:
                handle.remove()
        head = self.model.modality_heads[key](tokens, **inputs["head"])
        global_feat = self.model.modality_postprocessors[key](head).reshape(n, -1, head.size(-1)).mean(1)
        if layout == "none":
            return global_feat, None
        if key == "text":
            raise ValueError("Use layout none for ImageBind text")
        patches = (capture[0] if capture else tokens)[:, prep.num_cls_tokens:]
        if key == "imu":
            sequence = patches.reshape(n, -1, patches.size(-1))
        else:
            h, w = prep.patches_layout[-2:]
            grid = patches.reshape(n, -1, h, w, patches.size(-1))
            if layout == "spatial":
                return global_feat, spatial_pool(grid.mean(1), spec.get("grid_size", 14))
            if key == "audio":
                sequence = grid.mean(2).flatten(1, 2)
            else:
                sequence = grid.mean((2, 3))
        if layout != "temporal":
            raise ValueError(f"Unsupported {modality} layout {layout}")
        return global_feat, temporal_pool(sequence, spec.get("segments", 4))


def _load_array(root, value):
    return np.load(root / value, allow_pickle=False, mmap_mode="r") if isinstance(value, str) else value


def _extract_stream(encoder, spec, root, device, batch_size):
    source = spec["inputs"]
    rows = _load_array(root, source)
    if not len(rows):
        raise ValueError("A feature stream requires input rows")
    buffers = None
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        if isinstance(source, list):
            batch = np.stack([_load_array(root, path) for path in batch])
        inputs = torch.from_numpy(np.array(batch, copy=True)).to(device)
        with torch.inference_mode():
            features = encoder(spec, inputs)
        if buffers is None:
            buffers = [
                None if x is None else torch.empty((len(rows), *x.shape[1:]), dtype=torch.float32)
                for x in features
            ]
        for dest, feature in zip(buffers, features):
            if feature is None and dest is None:
                continue
            if dest is None or feature is None or feature.shape != dest[start:start + len(inputs)].shape:
                raise ValueError("The encoder returned inconsistent feature shapes")
            dest[start:start + len(inputs)].copy_(feature.detach())
    return tuple(buffers)


def _register(pack, local, options):
    ap, pb = split_disjoint_halves(pack)
    params = {"max_shift": 3, "max_frames": 2000, "seed": 0, "pool_size": 1, **options}
    pool = params.pop("pool_size")
    a, p, b = (local.get(key) for key in ("a", "p", "b"))
    pp = local.get("p_pb", p)
    if any(x is None or x.ndim != 4 for x in (a, p, pp, b)):
        raise ValueError("Registration requires a spatial grid on each fitting endpoint")

    def offset(left, right, indices):
        limit = params["max_frames"]
        if len(indices) > limit:
            rng = np.random.default_rng(params["seed"])
            indices = indices[np.sort(rng.choice(len(indices), limit, replace=False))]
        return estimate_spatial_offset(left[indices], right[indices], **params)[0]

    first, second = offset(a, p, ap), offset(pp, b, pb)
    local["a"], local["p"], local["b"] = align_spatial_grids(a, p, b, first, second, pool)
    if "p_pb" in local:
        local["p_pb"] = align_spatial_grids(a[:1], pp, b[:1], first, second, pool)[1]
    return {**params, "pool_size": pool, "offset_ap": list(first), "offset_pb": list(second)}


def extraction_cli(description, factory, argv=None):
    """Encode manifest streams in input order, preserving the pack's row namespaces."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    root = args.manifest.parent
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    streams = manifest.pop("streams")
    options = manifest.pop("encoder", {})
    registration = manifest.pop("registration", None)
    for key in ("train_idx", "test_idx", "ap_idx", "pb_idx", "db_idx", "labels", "groups", "coords", "db_coords"):
        if key in manifest:
            manifest[key] = _load_array(root, manifest[key])
    for key in ("query_sets", "query_coords"):
        if manifest.get(key) is not None:
            manifest[key] = {k: _load_array(root, v) for k, v in manifest[key].items()}
    encoder = factory(torch.device(device), options, root)
    local = {}
    for key, stream in streams.items():
        if key not in ("a", "p", "b", "p_pb"):
            raise ValueError(f"Unknown feature stream {key}")
        global_feat, local[key] = _extract_stream(encoder, stream, root, device, args.batch_size)
        field = "p_global_pb" if key == "p_pb" else f"{key}_global"
        manifest[field] = global_feat.numpy()
    if manifest["task"] == "wearable" and "a" not in streams:
        labels = np.asarray(manifest["labels"], dtype=np.int64)
        if not len(labels) or labels.min() < 0:
            raise ValueError("Wearable streams require nonnegative action labels")
        manifest["a_global"] = np.eye(int(labels.max()) + 1, dtype=np.float32)[labels]
    pack = RelationPack(**manifest)
    provenance = {
        "encoder": options,
        "streams": {k: {p: v for p, v in s.items() if p != "inputs"} for k, s in streams.items()},
    }
    if registration is not None:
        provenance["registration"] = _register(pack, local, registration)
    parts = {}
    for key, features in local.items():
        if features is not None:
            field = "p_local_pb" if key == "p_pb" else f"{key}_local"
            parts[field] = features.reshape(len(features), -1, features.size(-1)).numpy()
    pack = replace(pack, **parts, meta={**(pack.meta or {}), "extraction": provenance})
    path = save_relation_pack(args.output, pack, fp16=not args.fp32, compressed=args.compressed)
    print(path)
    return path


if __name__ == "__main__":
    extraction_cli(__doc__, ImageBindEncoder)
