"""Extract sensing features for localization and wearable recognition.

Run python scripts/extract_sensing.py --manifest relation.json --output pack.npz.
Use the split and stream manifest documented in extract_imagebind.py. Each
stream specifies encoder, modality, inputs and an optional layout. Encoder
choices are anythermal, dinov2, resnet18 and anymo. The first two use
Transformers checkpoints. ResNet18 uses torchvision ImageNet weights. AnyMo
uses its released raw-IMU pipeline and the checkpoint's custom Python code.

Visual arrays contain processor-normalized [N,3,H,W] images or [N,T,3,H,W]
clips. For a depth or thermal checkpoint expecting RGB-shaped input, prepare
three channels using that checkpoint's preprocessing. AnyMo inputs have shape
[N,T,S,6] with accelerometer followed by gyroscope channels. Supply the S
sensor_locations and the native sampling_rate in the IMU stream. A list of
per-sample NPY files and batch size one support variable-length recordings.

Use layout spatial for registered localization patches or temporal for action
segments. Layout none retains only the global feature. Wearable manifests may
omit stream a. Cross-device manifests use p_pb for the second RGB collection,
db_idx for A rows and query_sets for B rows. Dedicated coordinates follow
each selected index vector. Checkpoint IDs and revisions are configurable.
Model cards: https://huggingface.co/theairlabcmu/AnyThermal and
https://huggingface.co/CRUISEResearchGroup/AnyMo
"""

import torch
from torch import nn

if __package__:
    from .extract_imagebind import extraction_cli, spatial_pool, temporal_pool
else:
    from extract_imagebind import extraction_cli, spatial_pool, temporal_pool


_DEFAULTS = {
    "anythermal": "theairlabcmu/AnyThermal",
    "dinov2": "facebook/dinov2-base",
    "resnet18": "IMAGENET1K_V1",
    "anymo": "CRUISEResearchGroup/AnyMo",
}


class SensingEncoder:
    def __init__(self, device, options, root):
        self.device, self.options, self.root = device, options, root
        self.models = {}

    def load(self, name, spec):
        checkpoint = spec.get("checkpoint", _DEFAULTS[name])
        local = self.root / checkpoint
        if local.is_dir():
            checkpoint = str(local)
        revision = spec.get("revision", self.options.get("revision", "main"))
        key = (name, checkpoint, revision)
        if key in self.models:
            return self.models[key]
        if name == "anymo":
            from transformers import pipeline

            model = pipeline(
                "anymo", model=checkpoint, revision=revision, trust_remote_code=True,
                device=0 if self.device.type == "cuda" else -1, torch_dtype=torch.float32,
            )
            model.model.eval().requires_grad_(False)
        elif name == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            network = resnet18(weights=ResNet18_Weights[checkpoint])
            model = nn.Sequential(*list(network.children())[:-2]).to(self.device).eval().requires_grad_(False)
        else:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(checkpoint, revision=revision)
            model.to(self.device).eval().requires_grad_(False)
        self.models[key] = model
        return model

    def __call__(self, spec, value):
        name = spec["encoder"].lower()
        model = self.load(name, spec)
        layout = spec.get("layout", "temporal" if name == "anymo" or value.ndim == 5 else "spatial")
        segments = spec.get("segments", 4)
        if name == "anymo":
            if value.ndim != 4 or value.size(-1) != 6:
                raise ValueError("AnyMo requires [N,T,S,6] IMU input")
            locations, rate = spec["sensor_locations"], spec["sampling_rate"]
            if len(locations) != value.size(2) or rate <= 0:
                raise ValueError("Supply the sensor locations and a positive sampling rate")

            def encode(clip):
                result = model.encode_imu(clip.numpy(), locations, sampling_rate=rate)
                return torch.as_tensor(result).reshape(-1)

            clips = value.detach().float().cpu()
            global_feat = torch.stack([encode(clip) for clip in clips])
            if layout == "none":
                return global_feat, None
            if layout != "temporal" or not 1 <= segments <= clips.size(1):
                raise ValueError("AnyMo requires complete temporal segments")
            local = torch.stack([
                torch.stack([encode(part) for part in clip.tensor_split(segments, dim=0)])
                for clip in clips
            ])
            return global_feat, local

        if value.ndim == 4:
            value = value[:, None]
        if value.ndim != 5 or value.size(2) != 3:
            raise ValueError("Visual sensing input must contain three-channel frames")
        n, t = value.shape[:2]
        frames = value.flatten(0, 1).float()
        if name == "resnet18":
            grid = model(frames).permute(0, 2, 3, 1)
            per_frame = grid.mean((1, 2))
        else:
            hidden = model(pixel_values=frames).last_hidden_state
            step = model.config.patch_size
            h, w = frames.size(-2) // step, frames.size(-1) // step
            per_frame = hidden[:, 0]
            grid = hidden[:, -h * w:].reshape(n * t, h, w, -1)
        per_frame = per_frame.reshape(n, t, -1)
        global_feat = per_frame.mean(1)
        if layout == "none":
            return global_feat, None
        if layout == "temporal":
            return global_feat, temporal_pool(per_frame, segments)
        if layout == "spatial":
            grid = grid.reshape(n, t, *grid.shape[1:]).mean(1)
            return global_feat, spatial_pool(grid, spec.get("grid_size", 8))
        raise ValueError(f"Unknown sensing layout {layout}")


if __name__ == "__main__":
    extraction_cli(__doc__, SensingEncoder)
