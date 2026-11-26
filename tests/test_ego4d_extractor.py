"""CPU tests for the Ego4D protocol, preprocessing, and relation-pack extraction."""

import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import torch

from data import load_relation_pack
from evaluation import evaluate_relation
from pram import PRAMConfig
from scripts import extract_ego4d as extractor


_AXES = ("gyro_x", "gyro_y", "gyro_z", "accl_x", "accl_y", "accl_z")


def test_ego4d_protocol_schema_and_license_hygiene():
    path = Path(__file__).resolve().parents[1] / "configs" / "ego4d.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for task, size, n_train, n_test in (
        ("audio_imu", 1625, 1138, 487),
        ("text_imu", 2286, 1600, 686),
    ):
        spec = protocol[task]
        windows = spec["windows"]
        assert len(windows) == len(spec["labels"]) == len(spec["groups"]) == size
        assert spec["output_pack"] == f"ib_ego4d_{task}.npz"
        for window in windows:
            assert {"text", "narration_text", "scenario"}.isdisjoint(window)
            assert window["video_uid"]
            assert np.isfinite(window["timestamp_sec"])
            assert isinstance(window["ok"], bool)
            if task == "text_imu":
                assert window["narration_pass"] in {"narration_pass_1", "narration_pass_2", "missing"}
        train, test = spec["train_idx"], spec["test_idx"]
        assert len(train) == len(set(train)) == n_train
        assert len(test) == len(set(test)) == n_test
        assert set(train).isdisjoint(test)
        assert all(isinstance(i, int) and 0 <= i < size for i in train + test)


def test_narration_cleaning_and_lookup():
    clean = extractor.clean_narration
    fallback = extractor.FALLBACK_NARRATION
    assert clean("#C C washes the mug") == "The camera wearer washes the mug"
    assert clean("#C washes") == "washes"
    assert clean(" \n#c  c\t carries   blocks ") == "The camera wearer carries blocks"
    assert clean("#C closes the lid") == "closes the lid"
    assert clean(" \t ") == clean("#C ") == fallback

    first = {"narrations": [
        {"timestamp_sec": 2, "narration_text": "#C early"},
        {"timestamp_sec": 8, "narration_text": "#C C counts blocks"},
        {"narration_text": "Entry without a timestamp"},
    ]}
    second = {"narrations": [{"timestamp_sec": 7, "narration_text": "#C second pass"}]}
    narrations = {
        "both": {"narration_pass_1": first, "narration_pass_2": second},
        "first": {"narration_pass_1": first},
        "second": {"narration_pass_1": {"narrations": []}, "narration_pass_2": second},
        "empty": {},
    }
    lookup = extractor.lookup_narration
    assert lookup(narrations, "both", 7.1, "narration_pass_1") == "The camera wearer counts blocks"
    assert lookup(narrations, "both", 2.1, "narration_pass_2") == "second pass"
    assert lookup(narrations, "first", 2.1, "narration_pass_2") == "early"
    assert lookup(narrations, "second", 7, "narration_pass_1") == "second pass"
    assert lookup(narrations, "both", 2, "missing") == fallback
    assert lookup(narrations, "unknown", 2, "narration_pass_1") == fallback
    assert lookup(narrations, "empty", 2, "narration_pass_1") == fallback


def _write_imu(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*_AXES[::-1], "canonical_timestamp_ms"])
        writer.writeheader()
        for second, channels in samples:
            writer.writerow({"canonical_timestamp_ms": 1000 * second, **dict(zip(_AXES, channels))})


def test_imu_loading_and_5bin_pooling(tmp_path):
    path = tmp_path / "sensors.csv"
    _write_imu(path, [(s, [20 * axis + s for axis in range(6)]) for s in (10, 0, 4)])
    imu = extractor.load_imu_window(path, 5.0)
    assert imu.shape == (6, 2000)
    assert imu.dtype == torch.float32
    offsets = 20 * torch.arange(6, dtype=torch.float32)[:, None]
    expected = offsets + torch.linspace(0.0, 10.0, 2000)
    torch.testing.assert_close(imu, expected)
    for timestamp, edge in ((-100.0, 0), (100.0, 10)):
        outside = extractor.load_imu_window(path, timestamp)
        torch.testing.assert_close(outside, (offsets + edge).expand_as(imu))

    # A single impulse per slot makes the contribution of all 25 patches observable.
    channels = torch.arange(8, dtype=torch.float32)
    slot_means = torch.arange(10, dtype=torch.float32)[:, None].square() + channels
    patches = torch.zeros(10, 25, 8)
    patches[:, 11] = 25 * slot_means
    cls = torch.full((1, 8), 10000.0)
    tokens = torch.cat((cls, patches.reshape(250, 8))).unsqueeze(0)
    expected_bins = torch.stack([(slot_means[j] + slot_means[j + 1]) / 2 for j in range(2, 7)])
    for value in (tokens, tokens[0]):
        bins = extractor.imu_slots_to_5bins(value)
        assert bins.shape == (1, 5, 8)
        torch.testing.assert_close(bins[0], expected_bins, rtol=0, atol=0)


def _tiny_protocol():
    windows = [
        {"video_uid": f"clip-{i // 3}", "timestamp_sec": float(6 + 12 * (i % 3)), "ok": i != 10}
        for i in range(12)
    ]
    protocol = {}
    for task in ("audio_imu", "text_imu"):
        audio = task == "audio_imu"
        order = list(range(12)) if audio else list(reversed(range(12)))
        rows = [dict(windows[i]) for i in order]
        if not audio:
            for row in rows:
                row["timestamp_sec"] += 0.00000004
                row["narration_pass"] = "narration_pass_1"
        layers = {"audio": "head_cls", "imu": "block2_slots5"} if audio else {
            "text": "global", "imu": "block4_slots5",
        }
        protocol[task] = {
            "windows": rows, "output_pack": f"ib_ego4d_{task}.npz",
            "layers": {**layers, "video": "block24_cls"},
            "train_idx": [j for j, i in enumerate(order) if i < 8],
            "test_idx": [j for j, i in enumerate(order) if i >= 8],
            "labels": [i % 3 for i in order], "groups": [i // 3 for i in order],
        }
    return protocol


def _fake_feature(row, stream, shape):
    generator = torch.Generator(device="cpu").manual_seed(101 * stream + row)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


@pytest.fixture
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_check_inputs_and_end_to_end_extraction(tmp_path, monkeypatch, caplog, capsys, single_cpu_thread):
    root, output = tmp_path / "ego4d", tmp_path / "packs"
    protocol = _tiny_protocol()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    windows = protocol["audio_imu"]["windows"]
    row_ids = {(w["video_uid"], w["timestamp_sec"]): i for i, w in enumerate(windows)}

    video_dirs = [
        root / "video_540ss", root / "full_scale", root / "v2" / "video_540ss",
        root / "v2" / "full_scale",
    ]
    narrations = {}
    for group, directory in enumerate(video_dirs):
        uid = f"clip-{group}"
        directory.mkdir(parents=True)
        if group:
            (directory / f"{uid}.mp4").touch()
        recordings = [(w["timestamp_sec"], i) for i, w in enumerate(windows) if w["video_uid"] == uid]
        imu_dir = (root / "v2" if group % 2 else root) / "imu"
        _write_imu(imu_dir / f"{uid}.csv", [(s, [i] * 6) for s, i in recordings])
        narrations[uid] = {"narration_pass_1": {"narrations": [
            {"timestamp_sec": s, "narration_text": f"#C C counts marker {i}"} for s, i in recordings
        ]}}
    annotations = root / "annotations" / "narration.json"
    annotations.parent.mkdir()
    annotations.write_text(json.dumps(narrations), encoding="utf-8")

    cli = ["--ego4d_dir", str(root), "--protocol", str(protocol_path), "--output_dir", str(output),
           "--device", "cpu"]
    missing = video_dirs[0] / "clip-0.mp4"
    assert extractor.check_inputs(root, protocol)["missing"] == [str(missing)]
    with pytest.raises(SystemExit) as failure:
        extractor.main([*cli, "--check-inputs"])
    assert failure.value.code == 1
    assert missing.name in capsys.readouterr().out
    missing.touch()
    report = extractor.main([*cli, "--check-inputs"])
    assert report["missing_count"] == 0
    assert (report["video_count"], report["imu_count"], report["narration_count"]) == (4, 4, 1)
    assert not output.exists()

    absent = tmp_path / "absent"
    assert extractor.check_inputs(root, protocol, imu_dir=absent)["missing_count"] == 4
    assert extractor.check_inputs(root, protocol, video_dir=absent)["missing_count"] == 4
    assert extractor.check_inputs(root, protocol, narration_json=absent)["missing_count"] == 1
    assert extractor.check_inputs(root, protocol, ("audio_imu",), narration_json=absent)["missing_count"] == 0

    calls = Counter()
    real_imu_reader = extractor.load_imu_window

    def identify(path, timestamp):
        assert path.is_file()
        return row_ids[path.stem, round(timestamp, 6)]

    def video_reader(path, timestamp):
        i = identify(path, timestamp)
        calls["read_video", i] += 1
        if i == 10:
            raise OSError("Synthetic video decode failure")
        return torch.tensor(float(i)).expand(15, 3, 2, 224, 224)

    def audio_reader(path, timestamp):
        i = identify(path, timestamp)
        if i == 3:
            raise OSError("Synthetic audio decode failure")
        return torch.tensor(float(i)).expand(5, 1, 128, 204)

    def imu_reader(path, timestamp):
        calls["read_imu", identify(path, timestamp)] += 1
        return real_imu_reader(path, timestamp)

    class FakeRunner:
        def __init__(self, device, checkpoint=None):
            assert torch.device(device).type == "cpu"
            assert checkpoint is None

        def encode_video(self, clips):
            i = int(clips[0, 0, 0, 0, 0])
            calls["encode_video", i] += 1
            return _fake_feature(i, 1, (1024,)), _fake_feature(i, 2, (5, 1280))

        def encode_audio(self, mels):
            i = int(mels[0, 0, 0, 0])
            return _fake_feature(i, 3, (1024,)), _fake_feature(i, 4, (5, 768))

        def encode_imu(self, imu):
            assert imu.shape == (6, 2000)
            i = int(imu[0, 0])
            calls["encode_imu", i] += 1
            return _fake_feature(i, 5, (1024,)), _fake_feature(i, 6, (5, 512)), _fake_feature(i, 7, (5, 512))

        def encode_texts(self, texts):
            prefix = "The camera wearer counts marker "
            assert all(text.startswith(prefix) for text in texts)
            return torch.stack([_fake_feature(int(text[len(prefix):]), 8, (1024,)) for text in texts])

    monkeypatch.setattr(extractor, "load_video_clips", video_reader)
    monkeypatch.setattr(extractor, "load_audio_parts", audio_reader)
    monkeypatch.setattr(extractor, "load_imu_window", imu_reader)
    monkeypatch.setattr(extractor, "Ego4DImageBindRunner", FakeRunner)
    extractor.main([*cli, "--tasks", "audio_imu", "text_imu", "--compressed"])
    assert "Synthetic video decode failure" in caplog.text
    assert "Synthetic audio decode failure" in caplog.text
    for i in range(12):
        assert calls["read_video", i] == 1
        if i != 10:
            for operation in ("encode_video", "read_imu", "encode_imu"):
                assert calls[operation, i] == 1

    config = PRAMConfig(max_directions=3, knn_k=2, lbfgs_max_iter=4)
    for task, spec in protocol.items():
        audio = task == "audio_imu"
        order = list(range(12)) if audio else list(reversed(range(12)))
        failed = {3, 10} if audio else {10}
        path = output / spec["output_pack"]
        fields = {
            "a_global": (3 if audio else 8, (1024,)),
            "p_global": (1, (1024,)), "p_local": (2, (5, 1280)),
            "b_global": (5, (1024,)), "b_local": (6 if audio else 7, (5, 512)),
        }
        if audio:
            fields["a_local"] = (4, (5, 768))
        with ZipFile(path) as archive:
            assert all(entry.compress_type == ZIP_DEFLATED for entry in archive.infolist())
        with np.load(path, allow_pickle=False) as archive:
            assert ("a_local" in archive) == audio
            for field, (stream, shape) in fields.items():
                values = archive[field]
                dtype = np.float16 if field.endswith("_local") else np.float32
                assert values.dtype == dtype
                assert values.shape == (12, *shape)
                for row, i in enumerate(order):
                    if i in failed:
                        assert np.count_nonzero(values[row]) == 0
                    else:
                        expected = _fake_feature(i, stream, shape).numpy().astype(dtype)
                        np.testing.assert_array_equal(values[row], expected)

        pack = load_relation_pack(path)
        endpoint = "audio" if audio else "text"
        assert (pack.name, pack.backbone, pack.dataset) == (f"IB Ego4D {endpoint}-IMU", "ImageBind", "Ego4D")
        assert (pack.a_name, pack.b_name, pack.pivot, pack.metric, pack.task) == (
            endpoint, "IMU", "video", "instance", "main",
        )
        assert pack.meta == {"parts": "5 time segments" if audio else "text x 5 time segments", "source": "Ego4D v2"}
        assert (pack.a_local is None) == (not audio)
        for field, (_, shape) in fields.items():
            value = getattr(pack, field)
            assert value.shape == (12, *shape)
            assert value.dtype == np.float32
        for split in ("train_idx", "test_idx"):
            np.testing.assert_array_equal(getattr(pack, split), [j for j in spec[split] if order[j] not in failed])
        np.testing.assert_array_equal(pack.labels, spec["labels"])
        np.testing.assert_array_equal(pack.groups, spec["groups"])
        result = evaluate_relation(pack, config, seed=0, device="cpu")
        assert result["name"] == pack.name
        assert 0.0 <= result["retrieval"]["mean"]["R@1"] <= 100.0
        assert result["classification"] is not None

    precision_dir = tmp_path / "full_precision"
    extractor.main([*cli, "--output_dir", str(precision_dir), "--tasks", "text_imu", "--fp32"])
    with np.load(precision_dir / protocol["text_imu"]["output_pack"], allow_pickle=False) as archive:
        assert archive["p_local"].dtype == archive["b_local"].dtype == np.float32
