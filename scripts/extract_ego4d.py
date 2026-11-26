"""Extract PRAM relation packs from a licensed Ego4D v2 download.

Run ``python scripts/extract_ego4d.py --ego4d_dir ego4d --check-inputs``
to verify the download. Omit ``--check-inputs`` to extract features using
OpenCV, PyAV, and the official ImageBind package.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from contextlib import ExitStack
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(_ROOT))

from data import RelationPack, save_relation_pack


FALLBACK_NARRATION = "The camera wearer performs an activity."

_LOG = logging.getLogger(__name__)
_TASKS = ("audio_imu", "text_imu")
_IMU_COLUMNS = (
    "canonical_timestamp_ms", "gyro_x", "gyro_y", "gyro_z",
    "accl_x", "accl_y", "accl_z",
)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_GLOBAL = (1024,)
_VIDEO_LOCAL = (5, 1280)
_AUDIO_LOCAL = (5, 768)
_IMU_LOCAL = (5, 512)
_FATAL_ERRORS = (ImportError, MemoryError, torch.cuda.OutOfMemoryError)


def clean_narration(raw: str) -> str:
    """Expand the camera-wearer tag and normalize whitespace."""
    text = (raw or "").strip()
    text = re.sub(r"^#c\s+c\b", "The camera wearer", text, flags=re.IGNORECASE)
    text = re.sub(r"^#c(?:\s+|$)", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()) or FALLBACK_NARRATION


def lookup_narration(
    narrations: dict, video_uid: str, timestamp_sec: float, narration_pass: str,
) -> str:
    """Select the nearest timestamp within the first available narration pass."""
    if narration_pass == "missing" or video_uid not in narrations:
        return FALLBACK_NARRATION

    video = narrations[video_uid]
    passes = dict.fromkeys((narration_pass, "narration_pass_1", "narration_pass_2"))
    for name in passes:
        entries = (video.get(name) or {}).get("narrations") or []
        best = None
        best_distance = math.inf
        for entry in entries:
            try:
                distance = abs(float(entry.get("timestamp_sec")) - timestamp_sec)
            except (TypeError, ValueError):
                continue
            if distance < best_distance:
                best, best_distance = entry, distance
        if best is not None:
            return clean_narration(best.get("narration_text", ""))
    return FALLBACK_NARRATION


def imu_slots_to_5bins(tokens: torch.Tensor) -> torch.Tensor:
    """Pool ten one-second IMU slots into five centered temporal bins."""
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3:
        raise ValueError("IMU tokens require a sequence and a feature dimension")
    batch, length, width = tokens.shape
    if length <= 1 or (length - 1) % 10:
        raise ValueError("IMU patch tokens must divide into ten nonempty slots")
    if not tokens.is_floating_point():
        tokens = tokens.float()
    slots = tokens[:, 1:].reshape(batch, 10, (length - 1) // 10, width).mean(2)
    return 0.5 * (slots[:, 2:7] + slots[:, 3:8])


def _spatial_crops(rgb: np.ndarray) -> torch.Tensor:
    frame = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    height, width = frame.shape[-2:]
    short = min(height, width)
    size = (224 * height // short, 224 * width // short)
    frame = F.interpolate(
        frame.unsqueeze(0), size=size, mode="bilinear", align_corners=False,
    )[0]
    frame.sub_(frame.new_tensor(_CLIP_MEAN)[:, None, None])
    frame.div_(frame.new_tensor(_CLIP_STD)[:, None, None])
    dy, dx = size[0] - 224, size[1] - 224
    offsets = ((0, 0), ((dy + 1) // 2, (dx + 1) // 2), (dy, dx))
    return torch.stack([frame[:, y:y + 224, x:x + 224] for y, x in offsets])


def load_video_clips(video_path: Path, timestamp_sec: float) -> torch.Tensor:
    """Return three spatial crops for each of five two-frame clips."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError(f"Invalid frame rate or frame count in {video_path}")

        centers = np.clip(
            np.linspace(timestamp_sec - 2.0, timestamp_sec + 2.0, 5),
            0.0, max(0.0, count / fps - 0.01),
        )
        times = centers[:, None] + np.array([-0.5, 0.5])
        indices = np.clip(np.rint(times * fps), 0, count - 1).astype(np.int64)
        crops = {}
        for index in np.unique(indices):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                raise OSError(f"Cannot seek to frame {index} in {video_path}")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise OSError(f"Cannot read frame {index} in {video_path}")
            crops[index] = _spatial_crops(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    return torch.cat(
        [torch.stack((crops[left], crops[right]), dim=2) for left, right in indices],
        dim=0,
    )


def _resampled_frames(container, stream, resampler):
    for frame in container.decode(stream):
        yield from resampler.resample(frame)
    yield from resampler.resample(None)


def load_audio_parts(
    video_path: Path, timestamp_sec: float, window: float = 6.0, sr: int = 16000,
) -> torch.Tensor:
    """Decode a sample-aligned mono window and transform five overlapping clips."""
    import av
    from imagebind.data import waveform2melspec

    length = int(round(window * sr))
    if sr <= 0 or length < 1:
        raise ValueError("The audio window and sample rate must be positive")
    samples = np.zeros(length, dtype=np.float32)
    relative_start = max(0.0, timestamp_sec - window / 2)
    copied = False

    with av.open(str(video_path)) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream in {video_path}")
        stream = container.streams.audio[0]
        start = relative_start + (container.start_time or 0) / av.time_base
        first = round(start * sr)
        stop = first + length
        cursor = first
        if relative_start > 0:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
            cursor = None

        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sr)
        for frame in _resampled_frames(container, stream, resampler):
            if frame.pts is not None and frame.time_base is not None:
                cursor = round(frame.pts * frame.time_base * sr)
            if cursor is None:
                raise ValueError(f"Missing audio timestamp after seeking in {video_path}")
            end = cursor + frame.samples
            left, right = max(cursor, first), min(end, stop)
            if left < right:
                channel = frame.to_ndarray()[0]
                samples[left - first:right - first] = channel[left - cursor:right - cursor]
                copied = True
            if end >= stop:
                break
            cursor = end

    if not copied:
        raise ValueError(f"No audio samples in the requested window of {video_path}")
    waveform = torch.from_numpy(samples).unsqueeze(0)
    if length < 6 * sr:
        waveform = F.pad(waveform, (0, 6 * sr - length))
    mels = []
    for j in range(5):
        # ImageBind mean-centers each clip in place. Adjacent clips share samples.
        clip = waveform[:, j * sr:(j + 2) * sr].clone()
        mels.append(waveform2melspec(clip, sr, 128, 204))
    return (torch.stack(mels).float() + 4.268) / 9.138


@lru_cache(maxsize=2)
def _imu_samples(path: Path, _file_version: tuple[int, int]):
    """Reuse parsed CSV samples across nearby windows from the same recording."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(_IMU_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing IMU columns in {path}: {', '.join(sorted(missing))}")
        values = np.fromiter(
            (float(row[key] or "nan") for row in reader for key in _IMU_COLUMNS),
            dtype=np.float64,
        ).reshape(-1, len(_IMU_COLUMNS))
    values = values[np.isfinite(values[:, 0])]
    if not len(values):
        raise ValueError(f"No timestamped IMU samples in {path}")
    times, first = np.unique(values[:, 0], return_index=True)
    return times / 1000.0, values[first, 1:]


def load_imu_window(csv_path: Path, timestamp_sec: float, target: int = 2000) -> torch.Tensor:
    """Interpolate ten seconds of gyro-first IMU data onto a uniform time grid."""
    if target < 1:
        raise ValueError("The IMU sample count must be positive")
    path = Path(csv_path).resolve()
    stat = path.stat()
    times, axes = _imu_samples(path, (stat.st_mtime_ns, stat.st_size))
    start, end = timestamp_sec - 5.0, timestamp_sec + 5.0
    left = times.searchsorted(start, side="left")
    right = times.searchsorted(end, side="right")
    if left == right:
        nearest = np.argsort(np.abs(times - timestamp_sec), kind="stable")[:target]
        selection = np.sort(nearest)
    else:
        selection = slice(left, right)
    st, values = times[selection], axes[selection]
    grid = np.linspace(max(start, st[0]), min(end, st[-1]), target)
    channels = np.stack([np.interp(grid, st, channel) for channel in values.T])
    channels = channels.astype(np.float32)
    return torch.from_numpy(np.nan_to_num(channels, copy=False, posinf=0.0, neginf=0.0))


class Ego4DImageBindRunner:
    """Encode global embeddings and the task-specific frozen ImageBind layers."""

    def __init__(self, device: torch.device | str, checkpoint: Path | None = None):
        from imagebind.models.imagebind_model import imagebind_huge

        self.device = torch.device(device)
        self.model = imagebind_huge(pretrained=checkpoint is None)
        if checkpoint is not None:
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(weights)
        self.model.to(self.device).eval().requires_grad_(False)

    def _tower(self, modality, value, blocks=()):
        model = self.model
        inputs = model.modality_preprocessors[modality](**{modality: value.to(self.device)})
        trunk = model.modality_trunks[modality]
        tapped = {}

        def capture(_module, _args, output, *, block):
            tokens = output.transpose(0, 1).float()
            tapped[block] = F.layer_norm(tokens, (tokens.shape[-1],))

        with ExitStack() as hooks:
            for block in blocks:
                handle = trunk.blocks[block - 1].register_forward_hook(partial(capture, block=block))
                hooks.callback(handle.remove)
            tokens = trunk(**inputs["trunk"])

        head = model.modality_heads[modality](tokens, **inputs["head"])
        embedding = model.modality_postprocessors[modality](head)
        return F.normalize(embedding.float(), dim=-1), tokens, tapped

    @torch.inference_mode()
    def encode_video(self, clips: torch.Tensor):
        embedding, _, tapped = self._tower("vision", clips.float(), (24,))
        local = tapped[24][:, 0].reshape(5, 3, -1).mean(1)
        return embedding.mean(0), local

    @torch.inference_mode()
    def encode_audio(self, mels: torch.Tensor):
        embedding, tokens, _ = self._tower("audio", mels.float())
        local = self.model.modality_heads["audio"][0](tokens).float()[:, 0]
        return embedding[[0, 2, 4]].mean(0), local

    @torch.inference_mode()
    def encode_imu(self, imu: torch.Tensor):
        if imu.ndim == 2:
            imu = imu.unsqueeze(0)
        embedding, _, tapped = self._tower("imu", imu.float(), (2, 4))
        return embedding[0], imu_slots_to_5bins(tapped[2])[0], imu_slots_to_5bins(tapped[4])[0]

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        from imagebind.data import load_and_transform_text

        if not texts:
            return torch.empty((0, 1024), device=self.device, dtype=torch.float32)
        tokens = load_and_transform_text(texts, self.device)
        return self._tower("text", tokens)[0]


def _first_file(paths):
    return next((path for path in paths if path.is_file()), paths[0])


def _input_paths(ego4d_dir, protocol, tasks, video_dir=None, imu_dir=None, narration_json=None):
    root = Path(ego4d_dir)
    roots = (root, root / "v2")
    video_dirs = (
        [Path(video_dir)] if video_dir is not None else
        [base / folder for base in roots for folder in ("video_540ss", "full_scale")]
    )
    imu_dirs = [Path(imu_dir)] if imu_dir is not None else [base / "imu" for base in roots]
    uids = sorted({w["video_uid"] for task in tasks for w in protocol[task]["windows"]})
    media = {
        uid: (
            _first_file([directory / f"{uid}.mp4" for directory in video_dirs]),
            _first_file([directory / f"{uid}.csv" for directory in imu_dirs]),
        )
        for uid in uids
    }
    narration = None
    if "text_imu" in tasks:
        narration = (
            Path(narration_json) if narration_json is not None else
            _first_file([base / "annotations" / "narration.json" for base in roots])
        )
    return media, narration


def check_inputs(
    ego4d_dir: Path, protocol: dict, tasks=_TASKS, *,
    video_dir: Path | None = None, imu_dir: Path | None = None,
    narration_json: Path | None = None,
) -> dict:
    """Report missing files for the selected tasks without opening media or loading weights."""
    media, narration = _input_paths(ego4d_dir, protocol, tasks, video_dir, imu_dir, narration_json)
    required = [path for pair in media.values() for path in pair]
    if narration is not None:
        required.append(narration)
    missing = [str(path) for path in required if not path.is_file()]
    return {
        "video_count": len(media), "imu_count": len(media),
        "narration_count": int(narration is not None),
        "missing_count": len(missing), "missing": missing,
    }


def _feature(value, shape):
    array = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    if array.shape != shape:
        raise ValueError(f"Expected feature shape {shape}, received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("The encoder produced a nonfinite feature")
    return array


def _shared_features(runner, paths, timestamp):
    video_path, imu_path = paths
    video, video_local = runner.encode_video(load_video_clips(video_path, timestamp))
    imu, block2, block4 = runner.encode_imu(load_imu_window(imu_path, timestamp))
    return (
        _feature(video, _GLOBAL), _feature(video_local, _VIDEO_LOCAL),
        _feature(imu, _GLOBAL), _feature(block2, _IMU_LOCAL), _feature(block4, _IMU_LOCAL),
    )


def _fill_text_batch(runner, batch, output, valid):
    """Isolate a failing caption while retaining the other rows in its batch."""
    rows, texts = zip(*batch)
    try:
        output[list(rows)] = _feature(runner.encode_texts(list(texts)), (len(rows), 1024))
    except _FATAL_ERRORS:
        raise
    except Exception as exc:
        if len(batch) == 1:
            valid[rows[0]] = False
            _LOG.warning("Skipping text_imu row %d after a text encoding failure: %s", rows[0], exc)
            return
        middle = len(batch) // 2
        _fill_text_batch(runner, batch[:middle], output, valid)
        _fill_text_batch(runner, batch[middle:], output, valid)


def _extract_task(task, spec, media, narrations, runner, cache):
    audio = task == "audio_imu"
    windows = spec["windows"]
    count = len(windows)
    shapes = {
        "a_global": _GLOBAL, "p_global": _GLOBAL, "b_global": _GLOBAL,
        "p_local": _VIDEO_LOCAL, "b_local": _IMU_LOCAL,
    }
    if audio:
        shapes["a_local"] = _AUDIO_LOCAL
    features = {k: np.zeros((count, *shape), dtype=np.float32) for k, shape in shapes.items()}
    valid = np.zeros(count, dtype=bool)
    captions = []

    for row, window in enumerate(windows):
        uid = window["video_uid"]
        timestamp = float(window["timestamp_sec"])
        key = (uid, round(timestamp, 6))
        try:
            if key not in cache:
                # Keep a failed shared read from being retried by the other task.
                cache[key] = None
                cache[key] = _shared_features(runner, media[uid], timestamp)
            if cache[key] is None:
                raise ValueError("The shared video or IMU window failed on its first read")
            video, video_local, imu, block2, block4 = cache[key]
            if audio:
                embedding, local = runner.encode_audio(load_audio_parts(media[uid][0], timestamp))
                features["a_global"][row] = _feature(embedding, _GLOBAL)
                features["a_local"][row] = _feature(local, _AUDIO_LOCAL)
            else:
                caption = lookup_narration(narrations, uid, timestamp, window["narration_pass"])
                captions.append((row, caption))
            features["p_global"][row] = video
            features["p_local"][row] = video_local
            features["b_global"][row] = imu
            features["b_local"][row] = block2 if audio else block4
            valid[row] = True
        except _FATAL_ERRORS:
            raise
        except Exception as exc:
            _LOG.warning(
                "Skipping %s row %d (%s at %.6f s, protocol ok=%s): %s",
                task, row, uid, timestamp, window.get("ok", True), exc,
            )

    for start in range(0, len(captions), 32):
        _fill_text_batch(runner, captions[start:start + 32], features["a_global"], valid)
    if not valid.all():
        for array in features.values():
            array[~valid] = 0

    def kept(indices):
        indices = np.asarray(indices, dtype=np.int64)
        return indices[valid[indices]]

    endpoint = "audio" if audio else "text"
    return RelationPack(
        name=f"IB Ego4D {endpoint}-IMU", backbone="ImageBind", dataset="Ego4D",
        a_name=endpoint, b_name="IMU", pivot="video", metric="instance", task="main",
        train_idx=kept(spec["train_idx"]), test_idx=kept(spec["test_idx"]),
        labels=spec["labels"], groups=spec["groups"],
        meta={"parts": "5 time segments" if audio else "text x 5 time segments", "source": "Ego4D v2"},
        **features,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego4d_dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=_ROOT / "configs" / "ego4d.json")
    parser.add_argument("--output_dir", type=Path, default=Path("data_cache/packs/main"))
    parser.add_argument("--tasks", nargs="+", choices=_TASKS, default=list(_TASKS))
    parser.add_argument("--video_dir", type=Path)
    parser.add_argument("--imu_dir", type=Path)
    parser.add_argument("--narration_json", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    tasks = tuple(dict.fromkeys(args.tasks))
    overrides = dict(video_dir=args.video_dir, imu_dir=args.imu_dir, narration_json=args.narration_json)
    if args.check_inputs:
        report = check_inputs(args.ego4d_dir, protocol, tasks, **overrides)
        print(
            f"Checked {report['video_count']} videos, {report['imu_count']} IMU files, "
            f"and {report['narration_count']} narration files ({report['missing_count']} missing)."
        )
        if report["missing_count"]:
            for path in report["missing"][:5]:
                print(f"  {path}")
            raise SystemExit(1)
        return report

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    media, narration_path = _input_paths(args.ego4d_dir, protocol, tasks, **overrides)
    narrations = {}
    if narration_path is not None:
        with narration_path.open(encoding="utf-8") as handle:
            narrations = json.load(handle)
    runner = Ego4DImageBindRunner(device=device, checkpoint=args.checkpoint)
    cache = {}
    outputs = {}
    for task in tasks:
        spec = protocol[task]
        pack = _extract_task(task, spec, media, narrations, runner, cache)
        outputs[task] = save_relation_pack(
            args.output_dir / spec["output_pack"], pack,
            fp16=not args.fp32, compressed=args.compressed,
        )
        print(f"Saved {outputs[task]} ({len(pack.train_idx)} train, {len(pack.test_idx)} test)")
    return outputs


if __name__ == "__main__":
    main()
