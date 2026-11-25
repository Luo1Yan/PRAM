import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

_QUICKSTART_PATH = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
_QUICKSTART_SPEC = importlib.util.spec_from_file_location("pram_quickstart", _QUICKSTART_PATH)
assert _QUICKSTART_SPEC is not None and _QUICKSTART_SPEC.loader is not None
quickstart = importlib.util.module_from_spec(_QUICKSTART_SPEC)
sys.modules[_QUICKSTART_SPEC.name] = quickstart
_QUICKSTART_SPEC.loader.exec_module(quickstart)
from pram import (
    ModalityInput,
    PRAM,
    PRAMConfig,
    evaluate_bidirectional_retrieval,
    evaluate_classification,
    expected_recall_at_k,
)
from pram.mcr import fit_mcr
from pram.metrics import evaluate_bidirectional_knn
from pram.msf import _fit_pool
from pram.psrr import regularized_power


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
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


def test_disjoint_linear_gaussian_retrieval_and_classification():
    data = quickstart.make_benchmark(seed=14)
    pivots = torch.cat((data.ap_p.global_feat, data.pb_p.global_feat)).double()
    assert torch.unique(torch.round(pivots, decimals=4), dim=0).size(0) == len(pivots)
    results = quickstart.run_benchmark(data, "cpu")
    full = results["Full PRAM"]
    for key in ("a_to_b_r1", "b_to_a_r1", "accuracy"):
        assert full[key] > 0.75
        for baseline in ("Frozen Cosine", "Global-only PRAM"):
            assert full[key] > results[baseline][key] + 0.35


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

    data = quickstart.make_benchmark(9, parts=3)
    edges = (data.ap_a, data.ap_p, data.pb_p, data.pb_b)
    cfg = PRAMConfig(max_bank_size=17)
    standardized = PRAM(cfg).fit(*edges)
    raw = PRAM(replace(cfg, use_standardization=False)).fit(*edges)
    assert standardized.msf.w == 0
    left, right = data.test_a.take(slice(0, 8)), data.test_b.take(slice(3, 14))
    scores = raw.score_retrieval(left, right).double()
    ba = data.ap_a.take(standardized.msf.bank_indices_a)
    bb = data.pb_b.take(standardized.msf.bank_indices_b)
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


def test_quickstart_main_cpu(capsys):
    results = quickstart.main(["--device", "cpu"])
    output = capsys.readouterr().out
    for name in ("Frozen Cosine", "Global-only PRAM", "Full PRAM"):
        assert name in results and name in output
    assert results["Full PRAM"]["mean_r1"] > results["Global-only PRAM"]["mean_r1"] + 0.35
