import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from data import (
    RelationPack, load_relation_pack, resolve_pack_path, save_relation_pack, slice_modality,
    split_disjoint_halves, split_wearable_c3t, subsample_pair_budget,
)
from evaluation import evaluate_relation, evaluate_wearable_imu, localization_recall
from pram import (
    ModalityInput, PRAM, PRAMConfig, align_spatial_grids, estimate_spatial_offset,
    evaluate_bidirectional_retrieval, evaluate_classification, expected_recall_at_k,
)
from pram.mcr import fit_mcr
from pram.metrics import evaluate_bidirectional_knn
from pram.msf import _fit_pool
from pram.psrr import regularized_power
from run import main


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        assert not torch.cuda.is_available()
    yield
    torch.set_num_threads(previous)


def randn(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def draw(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    return draw


def state_edges():
    draw = randn(34)
    ya = torch.arange(4).repeat(12)
    yb = torch.tensor([0, 1, 1, 2, 4]).repeat(9)
    eye = torch.eye(5, dtype=torch.float64)
    pa, pb = eye[ya], eye[yb]
    a = pa.roll(1, dims=1)
    local = torch.stack((a, a.roll(1, dims=1), a.roll(2, dims=1)), dim=1)
    b = pb.roll(2, dims=1) + 0.035 * draw(len(yb), 5)
    return ModalityInput(a, local), ModalityInput(pa), ModalityInput(pb), ModalityInput(b)


@pytest.fixture(scope="module")
def state_model():
    edges = state_edges()
    model = PRAM(PRAMConfig(max_bank_size=23)).fit(*edges, is_class_prompt=True)
    return model, edges


def test_contraction_and_row_pooled_pivot_scale():
    draw = randn(11)
    na, nb, g, d = 37, 93, 3, 4
    pa = draw(na, g, d) * torch.tensor([0.3, 1.0, 2.0, 4.0]) + 13
    pb = draw(nb, g, d) * torch.tensor([2.5, 0.4, 1.2, 0.5]) - 21
    a = pa @ draw(d, 5) + 0.05 * draw(na, g, 5)
    b = pb @ draw(d, 6) + 0.05 * draw(nb, g, 6)

    def modality(x):
        return ModalityInput(x.mean(dim=1), x)

    cfg = PRAMConfig(relative_ridge=0.003)
    model = PRAM(cfg).fit(modality(a), modality(pa), modality(pb), modality(b))
    assert model.msf.omega == 1.0
    for spectral in model.psrr.groups:
        group = model.mcr.groups[spectral.group_index]
        cp_a = group.a.regularized.pp
        cp_b = group.b.regularized.pp
        assert not torch.allclose(cp_a, cp_b)
        vp = modality(pa).vectors(group.parts)
        vq = modality(pb).vectors(group.parts)
        pooled = (
            (vp - vp.mean(1, keepdim=True)).square().sum()
            + (vq - vq.mean(1, keepdim=True)).square().sum()
        ) / (vp.numel() + vq.numel())
        torch.testing.assert_close(group.a.p_scale.square(), pooled)
        torch.testing.assert_close(group.a.p_scale, group.b.p_scale)
        assert spectral.sigma.dtype == torch.float64
        assert spectral.sigma.max().item() <= 1 + 1e-5
        ca, cb = group.a.regularized, group.b.regularized
        h = (cp_a + cp_b) / 2
        ridge = cfg.relative_ridge * h.diagonal(dim1=1, dim2=2).mean(1)
        expected_k = ca.xp @ torch.linalg.solve(
            h + ridge[:, None, None] * torch.eye(d, dtype=h.dtype), cb.xp.mT
        )
        composed = regularized_power(ca.xx, -0.5, cfg) @ expected_k
        composed = composed @ regularized_power(cb.xx, -0.5, cfg)
        torch.testing.assert_close(spectral.sigma, torch.linalg.svdvals(composed))

    matrix = torch.diag(torch.tensor([-1e-9, 2.0], dtype=torch.float64))
    rho = cfg.relative_ridge * matrix.trace() / 2
    for power in (-0.5, -1.0):
        expected = (matrix.diagonal().clamp_min(0) + rho).pow(power)
        torch.testing.assert_close(regularized_power(matrix, power, cfg), expected.diag())


def test_shrinkage_excludes_broadcast_self_blocks():
    draw = randn(19)
    a, b = ModalityInput(draw(130, 5)), ModalityInput(draw(107, 6))
    scales = torch.tensor([0.15, 1.0, 3.0]).view(1, 3, 1)
    pa = ModalityInput(draw(len(a), 4), draw(len(a), 3, 4) * scales)
    pb = ModalityInput(draw(len(b), 4), draw(len(b), 3, 4) * scales)
    cfg = PRAMConfig()
    stats = fit_mcr(a, pa, pb, b, cfg)
    local = stats.groups[1]
    numerator = denominator = extra = 0.0
    for endpoint, pivot, edge in ((a, pa, local.a), (b, pb, local.b)):
        assert edge.x_broadcast and not edge.p_broadcast
        n = len(endpoint)
        x = (endpoint.vectors(3) - edge.x_mean[:, None]) / edge.x_scale
        p = (pivot.vectors(3) - edge.p_mean[:, None]) / edge.p_scale
        for left, right, weight, include in ((x, x, 1, False), (x, p, 2, True), (p, p, 1, True)):
            cov = left.mT @ right / n
            fourth = (left.square().sum(2) * right.square().sum(2)).mean(1)
            noise = (fourth - cov.square().sum((1, 2))).clamp_min(0).sum().item() / n
            if include:
                numerator += weight * noise
                denominator += weight * (cov - cov.mean(0)).square().sum().item()
            else:
                extra += noise
    assert 0 < stats.alpha < 1
    assert stats.alpha == pytest.approx(numerator / denominator)
    assert stats.alpha < min(1.0, (numerator + extra) / denominator)
    for edge in (local.a, local.b):
        expected = (1 - stats.alpha) * edge.sample.xp + stats.alpha * edge.sample.xp.mean(0)
        torch.testing.assert_close(edge.regularized.xp, expected)

    def identical(x):
        return ModalityInput(x.global_feat, x.global_feat[:, None].expand(-1, 3, -1))

    same = fit_mcr(a, identical(pa), identical(pb), b, cfg)
    assert same.alpha == 1.0
    off = fit_mcr(a, pa, pb, b, replace(cfg, use_shrinkage=False))
    assert off.alpha == 0.0
    torch.testing.assert_close(off.groups[1].a.sample.xp, off.groups[1].a.regularized.xp)


@pytest.mark.parametrize("states", [1, 3])
def test_single_vector_pivot_rank_and_rounding(states):
    draw = randn(42)
    centers = torch.eye(5, dtype=torch.float64)[:states]

    def pivot(n):
        if states == 1:
            return ModalityInput(torch.full((n, 5), 0.1, dtype=torch.float64))
        jitter = 1e-5 * draw(n, 5).tanh()
        return ModalityInput(centers[torch.arange(n) % states] + jitter)

    a = ModalityInput(draw(36, 7), draw(36, 2, 6))
    b = ModalityInput(draw(49, 8), draw(49, 2, 5))
    model = PRAM(PRAMConfig(use_pivot_associations=False)).fit(a, pivot(36), pivot(49), b)
    assert model.msf.m == states
    for group in model.psrr.groups:
        assert group.sigma.size(1) == max(1, states - 1)
        if states == 1:
            assert torch.count_nonzero(group.sigma) == 0
    if states == 1:
        assert torch.count_nonzero(model.score_retrieval(a, b)) == 0


def test_shared_states_class_propagation_and_fusion(state_model):
    model, (a, pa, pb, b) = state_model
    msf = model.msf
    assert msf.m == 5 and len(msf.shared_states) == 3
    assert msf.w == pytest.approx(1 - math.log(5) / math.log(len(a) + len(b)))
    assert msf.o == pytest.approx((3 / 4 + 4 / 5) / 2)
    assert msf.omega == pytest.approx(1 / 3)
    assert msf.covariance_maps and msf.pivot_route and msf.class_route
    assert msf.chi_A == pytest.approx(1.0)
    assert msf.w_A == pytest.approx(1 - math.log(4) / math.log(len(a)))
    assert msf.tau_A == pytest.approx(msf.w_A * msf.chi_A)
    assert len(msf.pivot_route.a.voters) == 2
    assert len(msf.pivot_route.b.voters) == 1
    torch.testing.assert_close(msf.pivot_route.a.weights.sum(), torch.tensor(1.0).double())

    ia, ib = msf.bank_indices_a, msf.bank_indices_b
    assert len(ia.unique()) == len(ib.unique()) == 23
    assert torch.equal(ia, torch.randperm(len(a), generator=torch.Generator().manual_seed(0))[:23])
    qa, qb = a.take(slice(0, 11)), b.take(slice(0, 13))

    def auxiliary(route, left, right):
        la, lb = route.a.log_posterior(left), route.b.log_posterior(right)
        return torch.logsumexp(la[:, None] + lb[None, :] - route.log_prior, dim=2)

    bank_a, bank_b = a.take(ia), b.take(ib)
    aux = []
    for route in (msf.pivot_route, msf.class_route):
        bank = auxiliary(route, bank_a, bank_b)
        torch.testing.assert_close(route.moments.mean, bank.mean())
        torch.testing.assert_close(route.moments.std, bank.std().clamp_min(1e-12))
        aux.append((auxiliary(route, qa, qb) - bank.mean()) / bank.std().clamp_min(1e-12))
    r = model.score_retrieval(qa, qb).double()
    s0 = (1 - msf.o) * r + msf.o * aux[0]
    expected = (1 - msf.tau_A) * s0 + msf.tau_A * aux[1]
    torch.testing.assert_close(model.score_classification(qa, qb), expected.float())

    no_aux = replace(model.config, use_pivot_associations=False, use_class_semantics=False)
    base = PRAM(no_aux).fit(a, pa, pb, b, is_class_prompt=True)
    assert base.msf.class_route is None and base.msf.pivot_route is None
    torch.testing.assert_close(base.score_classification(qa, qb), model.score_retrieval(qa, qb))
    plain = PRAM(model.config).fit(a, pa, pb, b)
    assert plain.msf.tau_A == 0 and plain.msf.class_route is None
    labels = pb.global_feat.argmax(1)
    torch.testing.assert_close(model.predict_knn(qa, b, labels), plain.predict_knn(qa, b, labels))
    torch.testing.assert_close(
        model.predict_class_prompts(qa, qb), model.score_classification(qa, qb).argmax(0)
    )


def test_pool_fallbacks_and_unseen_outer_states():
    cfg = PRAMConfig()
    draw = randn(7)
    z = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, -1.0]], dtype=torch.float64)
    classes = torch.arange(3)
    unique = ModalityInput(z, torch.stack((z, z), dim=1))
    pool = _fit_pool(unique, classes, cfg)
    torch.testing.assert_close(pool.weights, z.new_full((2,), 0.5))
    constant = ModalityInput(z.new_full((5, 2), 0.1))
    pool = _fit_pool(constant, torch.tensor([0, 0, 0, 1, 1]), cfg)
    expected = z.new_tensor([0.6, 0.4]).expand(5, -1)
    torch.testing.assert_close(pool.log_posterior(constant).exp(), expected)

    y = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    a = ModalityInput(z[y])
    p = ModalityInput(torch.eye(3, dtype=z.dtype)[y])
    model = PRAM(cfg).fit(a, p, p, ModalityInput(draw(len(y), 4)), is_class_prompt=True)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(7))
    left, right = order[:4], order[4:]
    correct = 0
    for train, test in ((left, right), (right, left)):
        fitted = _fit_pool(p.take(train), y[train], cfg)
        correct += (fitted.predict(p.take(test)) == y[test]).sum().item()
    expected_chi = max(0, (correct / len(y) - 1 / 3) / (1 - 1 / 3))
    assert model.msf.chi_A == pytest.approx(expected_chi)
    assert model.msf.chi_A < 1


@pytest.mark.parametrize("global_on,quadratic", [(True, True), (False, False)])
def test_raw_base_readout_and_component_ablations(global_on, quadratic):
    a, pa, pb, b = state_edges()
    cfg = PRAMConfig(
        use_global=global_on, use_quadratic=quadratic, use_standardization=False,
        use_pivot_associations=False,
    )
    model = PRAM(cfg).fit(a, pa, pb, b)
    qa, qb = a.take(slice(0, 7)), b.take(slice(0, 9))
    sc = torch.zeros(len(qa), len(qb), dtype=torch.float64)
    sv = torch.zeros_like(sc)
    ng = 0
    for spectral in model.psrr.groups:
        group = model.mcr.groups[spectral.group_index]
        assert global_on or group.parts > 0
        ac = qa.vectors(group.parts) - group.a.x_mean[:, None]
        bc = qb.vectors(group.parts) - group.b.x_mean[:, None]
        u = (ac / group.a.x_scale) @ spectral.projection_a
        v = (bc / group.b.x_scale) @ spectral.projection_b
        c = spectral.sigma.clamp(0, cfg.corr_cap)
        for g in range(len(c)):
            x, y = u[g, :, None], v[g, None, :]
            numerator = c[g] * x * y
            if quadratic:
                numerator -= 0.5 * c[g].square() * (x.square() + y.square())
            llr = (numerator / (1 - c[g].square())).sum(dim=2)
            sc += llr * (model.msf.omega if group.parts else 1.0)

        def mapped(endpoint, pivot, query):
            x = endpoint.vectors(group.parts)
            p = pivot.vectors(group.parts)
            x = x - x.mean(1, keepdim=True)
            p = p - p.mean(1, keepdim=True)
            xp = x.mT @ p / len(endpoint)
            pp = p.mT @ p / len(endpoint)
            phi = query @ xp @ regularized_power(pp, -0.5, cfg)
            return phi / phi.norm(dim=2, keepdim=True).clamp_min(cfg.scale_floor)

        ma, mb = mapped(a, pa, ac), mapped(b, pb, bc)
        sv += (ma @ mb.mT).sum(0)
        ng += len(c)
    assert ng == 3 + int(global_on)
    torch.testing.assert_close(model.score_retrieval(qa, qb), (sc + model.msf.w * sv / ng).float())
    no_cov = PRAM(replace(cfg, use_covariance_score=False)).fit(a, pa, pb, b)
    assert not no_cov.msf.covariance_maps
    torch.testing.assert_close(no_cov.score_retrieval(qa, qb), sc.float())


def test_reference_bank_invariance_and_sample_deviation(state_model):
    model, (a, _, _, b) = state_model
    for scorer in (model.score_retrieval, model.score_classification):
        reference = scorer(a, b)[7, 19]
        singleton = scorer(a.take(slice(7, 8)), b.take(slice(19, 20)))[0, 0]
        torch.testing.assert_close(singleton, reference, atol=1e-6, rtol=0)

    draw = randn(9)

    def modality(x):
        return ModalityInput(x.mean(dim=1), x)

    pa, pb = draw(35, 3, 4), draw(41, 3, 4)
    ap, bp = modality(pa + 0.1 * draw(*pa.shape)), modality(pb + 0.1 * draw(*pb.shape))
    edges = ap, modality(pa), modality(pb), bp
    cfg = PRAMConfig(max_bank_size=17)
    standardized = PRAM(cfg).fit(*edges)
    raw = PRAM(replace(cfg, use_standardization=False)).fit(*edges)
    assert standardized.msf.w == 0
    left, right = modality(draw(8, 3, 4)), modality(draw(11, 3, 4))
    scores = raw.score_retrieval(left, right).double()
    ba = ap.take(standardized.msf.bank_indices_a)
    bb = bp.take(standardized.msf.bank_indices_b)
    rows = raw.score_retrieval(left, bb).double()
    cols = raw.score_retrieval(ba, right).double()
    expected = 0.5 * (
        (scores - rows.mean(1, keepdim=True)) / rows.std(1, keepdim=True)
        + (scores - cols.mean(0)) / cols.std(0)
    )
    torch.testing.assert_close(standardized.score_retrieval(left, right), expected.float())


def test_chunked_scoring_and_single_row_banks(state_model):
    model, (a, _, _, b) = state_model
    assert model.to("cpu") is model
    for scorer in (model.score_retrieval, model.score_classification):
        full = scorer(a, b)
        assert full.dtype == torch.float32 and full.device.type == "cpu"
        assert (scorer(a, b, chunk_size=16) - full).abs().max().item() < 1e-5
        assert scorer(a.take(slice(0, 0)), b, chunk_size=16).shape == (0, len(b))
        with pytest.raises(ValueError):
            scorer(a, b, chunk_size=0)

    x = ModalityInput(torch.ones(1, 2))
    singleton = PRAM().fit(x, x, x, x, is_class_prompt=True)
    assert singleton.msf.w == singleton.msf.tau_A == 0
    assert singleton.score_retrieval(x, x).item() == 0
    assert singleton.score_classification(x, x).item() == 0


def test_instance_and_category_tie_expectations():
    scores = torch.tensor([[2, 2, 0], [4, 3, 3]])
    indices = torch.tensor([0, 2])
    assert expected_recall_at_k(scores, 1, indices) == pytest.approx(0.25)
    assert expected_recall_at_k(scores, 2, indices) == pytest.approx(0.75)
    assert expected_recall_at_k(scores, 0, indices) == 0
    assert expected_recall_at_k(scores, 10, indices) == 1

    scores = torch.tensor([[5, 4, 4, 4, 4, 2]])
    labels = dict(query_labels=torch.tensor([1]), candidate_labels=torch.tensor([0, 1, 1, 0, 0, 0]))
    for k, expected in ((1, 0), (2, 0.5), (3, 5 / 6), (4, 1)):
        assert expected_recall_at_k(scores, k, **labels) == pytest.approx(expected)
    labels["query_labels"] = torch.tensor([8])
    assert expected_recall_at_k(scores, 6, **labels) == 0

    tied = torch.ones(4, 4)
    result = evaluate_bidirectional_retrieval(tied, ks=(1, 3))
    for direction in ("a_to_b", "b_to_a", "mean"):
        assert result[direction] == {"R@1": 0.25, "R@3": 0.75}
    classes = torch.tensor([0, 0, 1, 1])
    category = evaluate_bidirectional_retrieval(tied, (2,), classes, classes)
    assert category["mean"]["R@2"] == pytest.approx(5 / 6)


def test_prompt_and_neighbor_tie_accuracy():
    scores = torch.tensor([[1, 1], [2, 1], [2, 1]])
    targets = torch.tensor([0, 0, 1])
    result = evaluate_classification(scores, targets)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.375)

    tied = torch.ones(2, 10)
    gallery = torch.tensor([3, 8]).repeat(5)
    targets = torch.tensor([3, 8])
    result = evaluate_classification(tied, targets, gallery, protocol="knn")
    assert result == {"accuracy": 0.5, "balanced_accuracy": 0.5}
    both = evaluate_bidirectional_knn(tied, tied, targets, targets, gallery, gallery)
    assert both["mean"] == result

    order = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    winner = gallery[order[0]].expand(2)
    result = evaluate_classification(
        tied, winner, gallery, protocol="knn", knn_k=1, max_gallery_size=3
    )
    assert result["accuracy"] == 1


def test_registration():
    a = randn(20)(31, 7, 7, 3)
    ap, pb = (1, -2), (-2, 1)
    p = a.roll(ap, dims=(1, 2))
    b = p.roll(pb, dims=(1, 2))
    for x, y, expected in ((a, p, ap), (p, b, pb)):
        offset, strengths = estimate_spatial_offset(x.flatten(1, 2), y, 2, 9, 3)
        assert offset == expected
        assert strengths.shape == (5, 5)
        assert strengths[expected[0] + 2, expected[1] + 2].item() == pytest.approx(1)
    aligned = align_spatial_grids(a, p, b, ap, pb, pool_size=2)
    expected = a[:, 1:5, 2:6].reshape(31, 2, 2, 2, 2, 3).mean((2, 4)).reshape(31, 4, 3)
    for x in aligned:
        torch.testing.assert_close(x, expected)
    with pytest.raises(ValueError):
        estimate_spatial_offset(a.flatten(1, 2)[:, :-1], p)
    with pytest.raises(ValueError):
        align_spatial_grids(a, p, b, (8, 0), pb)


@pytest.fixture
def pack():
    rng = np.random.default_rng(37)
    n = 50
    p = rng.normal(size=(n, 3, 4)).astype(np.float32)
    a = p + 0.1 * rng.normal(size=p.shape)
    b = p + 0.1 * rng.normal(size=p.shape)
    return RelationPack(
        name="aligned", dataset="fixture", backbone="frozen", task="main", metric="instance",
        a_name="depth", b_name="thermal", pivot="rgb",
        a_global=a.mean(1), p_global=p.mean(1), b_global=b.mean(1),
        a_local=a, p_local=p, b_local=b,
        train_idx=np.arange(36), test_idx=np.arange(36, n),
        ap_idx=np.arange(0, 36, 2)[::-1], pb_idx=np.arange(1, 36, 2)[::-1],
        labels=np.arange(n) % 4, coords=rng.normal(size=(n, 2)),
        groups=np.arange(n) // 5, class_names=["a", "b", "c", "d"], meta={"units": "meters"},
    )


def vehicle_pack(pack):
    return replace(
        pack, name="cross_device", task="vehicle",
        a_global=pack.a_global[:20], a_local=pack.a_local[:20],
        p_global=pack.p_global[:12], p_local=pack.p_local[:12],
        p_global_pb=pack.p_global[12:42], p_local_pb=pack.p_local[12:42],
        b_global=pack.b_global[12:], b_local=pack.b_local[12:],
        ap_idx=np.arange(12), pb_idx=np.arange(30),
        train_idx=np.array([], dtype=np.int64), test_idx=np.array([], dtype=np.int64),
        db_idx=np.arange(12, 20)[::-1], db_coords=pack.coords[12:20][::-1],
        query_sets={"map_night": np.arange(30, 34), "traversal_night": np.arange(34, 38)},
        query_coords={"map_night": pack.coords[42:46], "traversal_night": pack.coords[46:]},
        coords=None, groups=None, labels=None, class_names=None,
    )


def test_pack_io_and_splits(tmp_path, pack):
    path = save_relation_pack(tmp_path / "aligned.npz", pack, compressed=True)
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["a_global"].dtype == np.float32
        assert arrays["p_local"].dtype == np.float16
        assert arrays["train_idx"].dtype == arrays["labels"].dtype == np.int64
        assert arrays["coords"].dtype == np.float32
        assert json.loads(arrays["meta_json"].item())["name"] == pack.name
    loaded = load_relation_pack(path)
    np.testing.assert_array_equal(loaded.a_local, pack.a_local.astype(np.float16).astype(np.float32))
    assert loaded.p_global.dtype == np.float32
    assert loaded.meta == pack.meta and loaded.class_names == pack.class_names
    single = slice_modality(loaded.a_global, loaded.a_local[:, :1], loaded.test_idx[::-1])
    assert single.local_feat is None and single.global_feat.is_contiguous()

    for got, expected in zip(split_disjoint_halves(loaded), (loaded.ap_idx, loaded.pb_idx)):
        np.testing.assert_array_equal(got, np.sort(expected))
    order = np.random.default_rng(5).permutation(pack.train_idx)
    ap, pb = split_disjoint_halves(loaded, 5)
    np.testing.assert_array_equal(ap, np.sort(order[:18]))
    np.testing.assert_array_equal(pb, np.sort(order[18:]))
    rng = np.random.default_rng(6)
    for got, edge in zip(subsample_pair_budget(ap, pb, 0.25, 6), (ap, pb)):
        np.testing.assert_array_equal(got, np.sort(rng.choice(edge, round(len(edge) * 0.25), replace=False)))

    labels = np.repeat(np.arange(4), [11, 17, 19, 26])
    split = split_wearable_c3t(labels, 2)
    assert [len(split[k]) for k in ("test", "val", "har", "align")] == [7, 7, 29, 30]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(labels)))
    for key, rows in split.items():
        np.testing.assert_array_equal(rows, split_wearable_c3t(labels, 2)[key])
        assert set(labels[rows]) == set(labels)
    assert not np.array_equal(split["test"], split_wearable_c3t(labels, 3)["test"])

    separate = vehicle_pack(pack)
    saved = load_relation_pack(save_relation_pack(tmp_path / "vehicle.npz", separate, fp16=False))
    for got, expected in zip(split_disjoint_halves(saved, 4), (separate.ap_idx, separate.pb_idx)):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(saved.pb_pivot[1], separate.p_local_pb)
    for key in separate.query_sets:
        np.testing.assert_array_equal(saved.query_sets[key], separate.query_sets[key])
        np.testing.assert_array_equal(saved.query_coords[key], separate.query_coords[key])
    raw = asdict(separate)
    raw["p_global_pb"] = torch.from_numpy(raw["p_global_pb"]).half()
    torch.save(raw, tmp_path / "vehicle.pt")
    assert load_relation_pack(tmp_path / "vehicle.pt").p_global_pb.dtype == np.float32
    with pytest.raises(ValueError, match="outside"):
        replace(pack, test_idx=np.array([len(pack.a_global)]))
    with pytest.raises(ValueError, match="Missing required"):
        np.savez(tmp_path / "incomplete.npz", a_global=pack.a_global)
        load_relation_pack(tmp_path / "incomplete.npz")


def test_local_and_hub_resolution(tmp_path, monkeypatch):
    import huggingface_hub

    name = "packs/main/relation.npz"
    path = tmp_path / name
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(FileNotFoundError, match="--data_dir.*--hf_repo"):
        resolve_pack_path(name, tmp_path)
    assert resolve_pack_path(name, tmp_path, "owner/packs", "release") == path
    assert calls == [dict(repo_id="owner/packs", filename=name, repo_type="dataset",
                          revision="release", local_dir=str(tmp_path))]
    path.parent.mkdir(parents=True)
    path.touch()
    assert resolve_pack_path(name, tmp_path, "owner/packs") == path
    assert len(calls) == 1


def test_category_and_instance_cosine_protocols(pack):
    labels = pack.labels.copy()
    labels[pack.test_idx[-1]] = 4
    embeddings = np.eye(5, dtype=np.float32)[labels]
    category = replace(pack, labels=labels, metric="category", a_global=embeddings, b_global=embeddings)
    result = evaluate_relation(category, PRAMConfig(max_bank_size=10), chunk_size=6)
    assert result["frozen_cosine"]["classification"]["accuracy"] == 100
    assert result["frozen_cosine"]["retrieval"]["mean"]["R@1"] == 100

    labels[[1, 37]] = -1
    instance = replace(pack, labels=labels)
    cfg = PRAMConfig(max_gallery_size=9, knn_k=3)
    result = evaluate_relation(instance, cfg)
    assert result["gates"]["tau_A"] == 0
    gallery = instance.train_idx[labels[instance.train_idx] >= 0]
    gallery = np.sort(np.random.default_rng(0).choice(gallery, 9, replace=False))
    query = instance.test_idx[labels[instance.test_idx] >= 0]
    a, b = [torch.nn.functional.normalize(torch.from_numpy(x), dim=1) for x in (pack.a_global, pack.b_global)]
    targets, candidates = torch.tensor(labels[query]), torch.tensor(labels[gallery])
    expected = evaluate_bidirectional_knn(
        a[query] @ b[gallery].T, b[query] @ a[gallery].T, targets, targets,
        candidates, candidates, knn_k=3, max_gallery_size=9,
    )
    assert result["frozen_cosine"]["classification"]["mean"]["accuracy"] == pytest.approx(
        100 * expected["mean"]["accuracy"]
    )


@pytest.mark.parametrize("mode", [
    "main", "ablation", "sensing_robot", "sensing_vehicle", "sensing_wearable", "mechanism",
])
def test_run_modes(tmp_path, pack, mode, capsys):
    if mode == "sensing_vehicle":
        pack = vehicle_pack(pack)
    elif mode in ("sensing_robot", "sensing_wearable"):
        pack = replace(pack, task=mode.removeprefix("sensing_"))
    path = "packs/measurements.npz"
    save_relation_pack(tmp_path / path, pack)
    experiment = {
        "mode": mode, "relations": [{"name": "selected", "pack": path}, {"pack": "unselected.npz"}],
        "pram": {"knn_k": 3, "max_gallery_size": 20}, "variants": ["pram_full", "global_only"],
        "thresholds": [0.25, 5], "pair_budgets": [0.2, 1.0], "test_shuffled_pivot": True,
    }
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps(experiment), encoding="utf-8")
    override = tmp_path / "estimator.json"
    override.write_text('{"max_bank_size": 9}', encoding="utf-8")
    output = tmp_path / "reports" / "result.json"
    seeds = ["1", "2", "3"] if mode == "sensing_wearable" else ["0"]
    report = main([
        "--config", str(config), "--pram_config", str(override), "--data_dir", str(tmp_path),
        "--relations", "selected", "--device", "cpu", "--chunk_size", "5",
        "--output", str(output), "--seeds", *seeds,
    ])
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["device"] == "cpu" and report["mode"] == mode
    assert report["pram"]["max_bank_size"] == 9 and report["pram"]["knn_k"] == 3
    assert "selected\t" in capsys.readouterr().out
    row, = report["results"]
    if mode == "main":
        assert set(row["retrieval"]["mean"]) == {"R@1", "R@5", "R@10"}
        assert set(row["gates"]) == {"alpha", "omega", "w", "o", "tau_A"}
        assert report["means"][pack.backbone]["pram_full"]["R@10"] == row["retrieval"]["mean"]["R@10"]
    elif mode == "ablation":
        assert set(row["variants"]) == {"pram_full", "global_only"}
        assert row["frozen_cosine"]["classification"]["mean"]["accuracy"] >= 0
    elif mode == "sensing_robot":
        assert set(row["thermal_to_depth"]) == {"0.25m", "5m"}
        assert set(row["exact_retrieval"]["mean"]) == {"R@1", "R@10"}
    elif mode == "sensing_vehicle":
        assert set(row["query_sets"]) == {"map_night", "traversal_night"}
        assert all(0 <= values["5m"] <= 100 for values in row["query_sets"].values())
    elif mode == "sensing_wearable":
        assert set(row["per_seed"]) == {"1", "2", "3"}
        assert 0 <= row["mean"]["Top-1"] <= row["mean"]["Top-3"] <= 100
        assert row["mean"]["Top-1"] == pytest.approx(np.mean([s["Top-1"] for s in row["per_seed"].values()]))
    else:
        assert set(row["pair_budgets"]) == {"0.2", "1.0"}
        assert row["pair_budgets"]["0.2"]["0"]["wo_shrinkage"]["gates"]["alpha"] == 0
        assert set(row["shuffled_pivot"]) == {"aligned", "shuffled"}


def test_wearable_fitting_rows_and_tied_ranking(pack, monkeypatch):
    rows = np.arange(len(pack.labels), dtype=np.float32)[:, None]
    wearable = replace(pack, task="wearable", p_global=rows, b_global=rows, p_local=None, b_local=None)
    split = split_wearable_c3t(pack.labels, 1)

    def fit(self, a, pa, pb, b, is_class_prompt=False):
        assert not is_class_prompt and a.local_feat is None
        np.testing.assert_array_equal(pa.global_feat[:, 0].numpy(), split["har"])
        np.testing.assert_array_equal(a.global_feat.argmax(1).numpy(), pack.labels[split["har"]])
        for edge in (pb, b):
            np.testing.assert_array_equal(edge.global_feat[:, 0].numpy(), split["align"])
        return self

    def score(self, a, b, chunk_size=None):
        torch.testing.assert_close(a.global_feat, torch.eye(4))
        np.testing.assert_array_equal(b.global_feat[:, 0].numpy(), split["test"])
        return torch.zeros(4, len(b))

    monkeypatch.setattr(PRAM, "fit", fit)
    monkeypatch.setattr(PRAM, "score_classification", score)
    result = evaluate_wearable_imu(wearable, PRAMConfig(), [1])
    labels = wearable.labels[split["test"]]
    assert result["mean"]["Top-1"] == pytest.approx(100 * np.mean(labels == 0))
    assert result["mean"]["Top-3"] == pytest.approx(100 * np.mean(labels < 3))


def test_localization_coordinates_and_cutoff():
    scores = torch.tensor([[1, 2, 0], [3, 1, 2]])
    database = np.array([[10, 0], [0, 0], [30, 0]])
    queries = np.array([[3, 4], [10, 0]])
    assert localization_recall(scores, queries, database, [0, 4, 5]) == {
        "0m": 50.0, "4m": 50.0, "5m": 100.0,
    }


def test_auto_device_with_cuda_hidden(tmp_path, pack):
    save_relation_pack(tmp_path / "relation.npz", pack)
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"mode": "main", "relations": ["relation.npz"]}), encoding="utf-8")
    output = tmp_path / "report.json"
    code = (
        "import runpy, torch\n"
        "assert not torch.cuda.is_available()\n"
        "torch.set_num_threads(1)\n"
        "runpy.run_module('run', run_name='__main__')\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", code, "--config", str(config), "--data_dir", str(tmp_path),
         "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, capture_output=True, text=True,
        check=True, timeout=90,
    )
    assert "relation\t" in child.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["device"] == "cpu"


def test_shared_continuous_pivots_preserve_classification():
    draw = randn(51)
    centers = torch.eye(4, dtype=torch.float64)
    signal = centers.repeat_interleave(4, dim=0)
    nuisance = 4 * centers.repeat(4, 1)
    # Opposite perturbations keep the pivot uncorrelated with the nuisance features.
    jitter = 0.05 * draw(*signal.shape)
    p = ModalityInput(torch.cat((signal + jitter, signal - jitter)))
    a = ModalityInput(torch.cat((signal, nuisance), dim=1).repeat(2, 1))
    b = ModalityInput(torch.cat((nuisance, signal), dim=1).repeat(2, 1))
    model = PRAM().fit(a, p, p, b, is_class_prompt=False)
    assert model.msf.m == len(p)
    assert model.msf.o == 0.0
    assert model.msf.pivot_route is None

    prompts = ModalityInput(torch.cat((centers, 4 * centers), dim=1))
    labels = torch.arange(4).repeat(6)
    query_signal = centers[labels] + 0.02 * draw(len(labels), 4)
    queries = ModalityInput(torch.cat((4 * centers[(labels + 1) % 4], query_signal), dim=1))
    scores = model.score_classification(prompts, queries)
    torch.testing.assert_close(scores, model.score_retrieval(prompts, queries), rtol=0, atol=0)
    assert evaluate_classification(scores.T, labels)["accuracy"] > 0.9


def test_half_overlapping_continuous_pivots():
    frames = randn(52)(36, 4)
    pa, pb = ModalityInput(frames[:24]), ModalityInput(frames[12:])
    msf = PRAM().fit(pa, pa, pb, pb).msf
    assert msf.m == len(frames)
    assert msf.o == 0.0
    assert msf.pivot_route is None


def test_mixed_pivots_keep_repeated_states():
    frames = randn(53)(30, 2)
    states = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    pa = ModalityInput(torch.cat((frames, states[:1], states[1:].repeat(11, 1))))
    pb = ModalityInput(torch.cat((frames, states[:1].repeat(5, 1), states[1:].repeat(7, 1))))
    msf = PRAM().fit(pa, pa, pb, pb).msf
    assert len(msf.shared_states) == 2
    assert msf.o == pytest.approx(12 / 42)
    assert msf.pivot_route is not None
    torch.testing.assert_close(msf.shared_states, states)
    torch.testing.assert_close(msf.pivot_route.log_prior.exp(), states.new_tensor([0.25, 0.75]))


def test_incidental_pivot_singletons_preserve_overlap():
    draw = randn(54)
    shared = draw(3, 4)
    pa = ModalityInput(torch.cat((shared, draw(37, 4))))
    pb = ModalityInput(torch.cat((shared, shared[2:].repeat(2, 1), draw(35, 4))))
    msf = PRAM().fit(pa, pa, pb, pb).msf
    assert len(msf.shared_states) == 3
    assert msf.o == 0.5 * (3 / 40 + 5 / 40)
    assert msf.pivot_route is not None


def test_shared_continuous_pivots_preserve_class_prompt_support():
    centers = torch.eye(16, dtype=torch.float64)
    labels = torch.cat((torch.zeros(12, dtype=torch.long), torch.arange(1, 16)))
    a = ModalityInput(centers[labels])
    p = ModalityInput(a.global_feat + 0.02 * randn(55)(len(a), len(centers)))
    b = ModalityInput(p.global_feat.roll(1, dims=1))
    msf = PRAM().fit(a, p, p, b, is_class_prompt=True).msf
    assert msf.o == 0.0
    assert msf.pivot_route is None
    assert msf.tau_A > 0
    assert msf.class_route is not None
    assert len(msf.class_route.a.classes) == len(centers)
