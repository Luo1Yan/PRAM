"""Portable numeric feature packs. All coordinate arrays use local meters."""

from __future__ import annotations

import json
import os
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from pram import ModalityInput


_TEXT = ("name", "backbone", "dataset", "a_name", "b_name", "pivot", "metric", "task")
_GLOBAL = ("a_global", "p_global", "b_global", "p_global_pb")
_LOCAL = ("a_local", "p_local", "b_local", "p_local_pb")
_INDICES = ("train_idx", "test_idx", "ap_idx", "pb_idx", "db_idx")
_QUERY = ("query_sets", "query_coords")


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def _float(value, rank, name):
    a = _array(value)
    if a.ndim != rank or a.dtype.kind not in "fiu" or min(a.shape[1:]) < 1:
        raise ValueError(f"{name} requires a real rank-{rank} array with positive width")
    a = np.ascontiguousarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return a


def _integer(value, name):
    a = _array(value)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"{name} requires an integer vector")
    return np.ascontiguousarray(a, dtype=np.int64)


def _bounds(a, size, name):
    if a is not None and a.size and (a.min() < 0 or a.max() >= size):
        raise ValueError(f"{name} contains a row outside [0, {size})")


def _xy(value, name):
    a = _float(value, 2, name)
    if a.shape[1] != 2:
        raise ValueError(f"{name} requires two coordinate columns")
    return a


@dataclass
class RelationPack:
    """Rows on each fitting edge share an index within that edge's storage."""

    name: str
    backbone: str
    dataset: str
    a_name: str
    b_name: str
    pivot: str
    metric: str
    task: str
    a_global: np.ndarray
    p_global: np.ndarray
    b_global: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    a_local: np.ndarray | None = None
    p_local: np.ndarray | None = None
    b_local: np.ndarray | None = None
    ap_idx: np.ndarray | None = None
    pb_idx: np.ndarray | None = None
    p_global_pb: np.ndarray | None = None
    p_local_pb: np.ndarray | None = None
    labels: np.ndarray | None = None
    groups: np.ndarray | None = None
    coords: np.ndarray | None = None
    db_idx: np.ndarray | None = None
    db_coords: np.ndarray | None = None
    query_sets: dict[str, np.ndarray] | None = None
    query_coords: dict[str, np.ndarray] | None = None
    class_names: list[str] | None = None
    meta: dict | None = None

    @property
    def separate_edges(self) -> bool:
        sizes = {len(self.a_global), len(self.p_global), len(self.b_global)}
        return (
            self.p_global_pb is not None or self.p_local_pb is not None or len(sizes) > 1
            or self.task == "vehicle" or bool((self.meta or {}).get("separate_collections"))
        )

    @property
    def pb_pivot(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self.p_global_pb is not None:
            return self.p_global_pb, self.p_local_pb
        local = self.p_local if self.p_local_pb is None else self.p_local_pb
        return self.p_global, local

    def __post_init__(self):
        for key in _TEXT:
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} requires a nonempty string")
            setattr(self, key, value)
        if self.metric not in ("instance", "category"):
            raise ValueError("metric must be instance or category")
        if self.task not in ("main", "robot", "vehicle", "wearable"):
            raise ValueError("Unknown relation task")
        if self.meta is not None and not isinstance(self.meta, dict):
            raise ValueError("meta must be a JSON object")
        if self.class_names is not None:
            if isinstance(self.class_names, str) or not all(
                isinstance(name, str) for name in self.class_names
            ):
                raise ValueError("class_names must contain class strings")
            self.class_names = list(self.class_names)

        for key in (*_GLOBAL, *_LOCAL):
            value = getattr(self, key)
            if value is not None or key in _GLOBAL[:3]:
                setattr(self, key, _float(value, 2 if key in _GLOBAL else 3, key))
        pa, pb = self.p_global, self.pb_pivot[0]
        if pa.shape[1] != pb.shape[1]:
            raise ValueError("The two global pivot widths must agree")
        globals_ = (self.a_global, pa, self.b_global, pb)
        slots = set()
        for key, global_feat in zip(_LOCAL, globals_):
            local = getattr(self, key)
            if local is not None:
                if len(local) != len(global_feat):
                    raise ValueError(f"{key} and its global features must share rows")
                if local.shape[1] > 1:
                    slots.add(local.shape[1])
        if len(slots) > 1:
            raise ValueError("All multi-part local features must share a part count")
        if slots:
            def pivot_width(global_feat, local):
                return local.shape[2] if local is not None and local.shape[1] > 1 else global_feat.shape[1]

            if pivot_width(pa, self.p_local) != pivot_width(*self.pb_pivot):
                raise ValueError("The two local pivot widths must agree")

        for key in (*_INDICES, "labels"):
            value = getattr(self, key)
            if value is not None or key in ("train_idx", "test_idx"):
                vector = _integer(value, key)
                if key in _INDICES and len(np.unique(vector)) != len(vector):
                    raise ValueError(f"{key} contains duplicate rows")
                setattr(self, key, vector)
        na, nb = len(self.a_global), len(self.b_global)
        _bounds(self.ap_idx, min(na, len(pa)), "ap_idx")
        _bounds(self.pb_idx, min(nb, len(pb)), "pb_idx")
        _bounds(self.db_idx, na, "db_idx")
        if self.separate_edges:
            _bounds(self.train_idx, max(na, nb, len(pa), len(pb)), "train_idx")
        else:
            _bounds(self.train_idx, len(pa), "train_idx")
            if self.ap_idx is not None and self.pb_idx is not None:
                if np.intersect1d(self.ap_idx, self.pb_idx).size:
                    raise ValueError("Shared-index fitting edges must be disjoint")
        if self.task != "vehicle":
            _bounds(self.test_idx, min(na, nb), "test_idx")
            for idx in (self.train_idx, self.ap_idx, self.pb_idx):
                if idx is not None and np.intersect1d(idx, self.test_idx).size:
                    raise ValueError("Fitting rows overlap evaluation rows")
        if self.groups is not None:
            groups = _array(self.groups)
            if groups.ndim != 1 or groups.dtype.kind not in "iuUS":
                raise ValueError("groups requires a vector of integer or string identifiers")
            self.groups = groups.astype(str) if groups.dtype.kind in "US" else _integer(groups, "groups")
        if self.coords is not None:
            self.coords = _xy(self.coords, "coords")
        if not self.separate_edges:
            for key in ("labels", "groups", "coords"):
                value = getattr(self, key)
                if value is not None and len(value) != na:
                    raise ValueError(f"{key} must have one entry per aligned row")
        if self.task == "wearable" and not (len(pa) == nb and (
            self.labels is None or len(self.labels) == nb
        )):
            raise ValueError("Wearable labels, video and IMU must share rows")
        if self.query_sets is not None:
            self.query_sets = {k: _integer(v, "query_sets") for k, v in self.query_sets.items()}
            for key, rows in self.query_sets.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("Query sets require nonempty string names")
                _bounds(rows, nb, key)
        if self.db_coords is not None:
            self.db_coords = _xy(self.db_coords, "db_coords")
            if self.db_idx is None or len(self.db_coords) not in (len(self.db_idx), na):
                raise ValueError("db_coords must describe database rows or all A rows")
        if self.query_coords is not None:
            self.query_coords = {k: _xy(v, "query_coords") for k, v in self.query_coords.items()}
            for key, xy in self.query_coords.items():
                if self.query_sets is None or key not in self.query_sets:
                    raise ValueError(f"Missing query set for coordinates {key}")
                if len(xy) not in (len(self.query_sets[key]), nb):
                    raise ValueError(f"Coordinate count differs from query set {key}")


def save_relation_pack(
    path: str | Path, pack: RelationPack, fp16: bool = True, compressed: bool = False,
) -> Path:
    """Write one numeric NPZ archive with a UTF-8 JSON header and named query arrays."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    if path.suffix.lower() != ".npz":
        raise ValueError("Feature archives must use the .npz extension")
    pack = replace(pack)
    header = {key: getattr(pack, key) for key in (*_TEXT, "meta", "class_names")}
    arrays = {}
    with np.errstate(over="raise", invalid="raise"):
        for key in (*_GLOBAL, *_LOCAL, *_INDICES, "labels", "groups", "coords", "db_coords"):
            value = getattr(pack, key)
            if value is not None:
                if key in _GLOBAL:
                    value = value.astype(np.float32, copy=False)
                elif key in _LOCAL:
                    value = value.astype(np.float16 if fp16 else np.float32, copy=False)
                arrays[key] = value
    for key in _QUERY:
        mapping = getattr(pack, key)
        if mapping is not None:
            header[key] = {}
            for i, (name, value) in enumerate(mapping.items()):
                member = f"{key}_{i}"
                header[key][name] = member
                arrays[member] = value
    arrays["meta_json"] = np.asarray(json.dumps(header, ensure_ascii=False, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _decode(raw):
    header = {}
    if "meta_json" in raw:
        header = json.loads(_array(raw["meta_json"]).item())
        if not isinstance(header, dict):
            raise ValueError("meta_json must encode an object")
    values = {}
    for field in fields(RelationPack):
        key = field.name
        if key in raw:
            value = raw[key]
        elif key in header:
            value = header[key]
        elif field.default is MISSING:
            raise ValueError(f"Missing required pack field {key}")
        else:
            continue
        if key in _QUERY and value is not None:
            value = {k: raw[v] if isinstance(v, str) else v for k, v in value.items()}
        values[key] = value
    return RelationPack(**values)


def load_relation_pack(path: str | Path) -> RelationPack:
    """Load a pickle-free NPZ or a trusted Torch pack onto CPU in float32."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return _decode(archive)
    if path.suffix.lower() == ".pt":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, RelationPack):
            return replace(raw)
        if isinstance(raw, dict):
            return _decode(raw)
        raise ValueError("A Torch pack must contain a relation dictionary or RelationPack")
    raise ValueError("Supported feature-pack extensions are .npz and .pt")


def slice_modality(
    global_arr: np.ndarray, local_arr: np.ndarray | None, indices: np.ndarray | slice,
) -> ModalityInput:
    """Select contiguous float32 rows, broadcasting single vectors in the estimator."""
    def select(value):
        return torch.from_numpy(np.ascontiguousarray(value[indices], dtype=np.float32))

    local = None if local_arr is None or local_arr.shape[1] == 1 else select(local_arr)
    return ModalityInput(select(global_arr), local)
