#!/usr/bin/env python3
"""Small analytic conditional-field fixture; no cosmological source data.

This tests the diagnostic machinery, not field-model calibration or statistical
power. A 12-cell Gaussian field is observed at four cells with known noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def gaussian_condition(covariance, observation_matrix, noise_covariance, observation):
    system = observation_matrix @ covariance @ observation_matrix.T + noise_covariance
    gain = np.linalg.solve(system, observation_matrix @ covariance).T
    mean = gain @ observation
    posterior_covariance = covariance - gain @ observation_matrix @ covariance
    return mean, (posterior_covariance + posterior_covariance.T) / 2


def mira_per_parent(truth, draws, centers):
    """One independent boundary draw; all other draws count region mass.

    The final draw is excluded from the mass count. Rows are parent examples,
    never treated as independent just because several regions are drawn.
    """
    if draws.shape[1] < 2:
        raise ValueError("MIRA needs a mass sample and an independent boundary draw")
    count = draws.shape[1] - 1
    radius2 = np.sum((draws[:, -1] - centers) ** 2, axis=-1)
    inside = np.sum((draws[:, :-1] - centers[:, None]) ** 2, axis=-1) <= radius2[:, None]
    n = inside.sum(axis=1)
    k = np.sum((truth - centers) ** 2, axis=-1) <= radius2
    return np.where(k, n + 1, count - n + 1) / (count + 2)


def make_fixture(parents=128, draws=32, seed=20260905):
    if not 4 <= parents <= 256 or not 4 <= draws <= 128:
        raise ValueError("fixture exceeds its small analytic budget")
    rng = np.random.default_rng(seed)
    positions = np.stack(np.unravel_index(np.arange(12), (2, 2, 3)), axis=1)
    distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    covariance = 0.7 * np.exp(-distance / 2) + 0.3 * np.eye(12)
    observation_matrix = np.eye(12)[[0, 3, 5, 8]]
    truth = rng.multivariate_normal(np.zeros(12), covariance, size=parents)
    sigma = np.where(np.arange(parents) % 2, 1.0, 0.25)
    observations = truth @ observation_matrix.T + rng.normal(size=(parents, 4)) * sigma[:, None]
    means, covariances = [], []
    for i in range(parents):
        mu, cov = gaussian_condition(covariance, observation_matrix, sigma[i] ** 2 * np.eye(4), observations[i])
        means.append(mu)
        covariances.append(cov)
    means, covariances = np.array(means), np.array(covariances)
    eps = rng.normal(size=(parents, draws, 12))
    correct = means[:, None] + np.einsum("pij,pmj->pmi", np.linalg.cholesky(covariances), eps)
    broken_cov = covariances.copy()
    broken_cov[:, :6, 6:] = 0
    broken_cov[:, 6:, :6] = 0
    independent = means[:, None] + np.einsum("pij,pmj->pmi", np.linalg.cholesky(broken_cov), eps)
    prior = np.einsum("ij,pmj->pmi", np.linalg.cholesky(covariance), eps)
    candidates = {
        "exact_posterior": correct,
        "ignores_observation_prior": prior,
        "independent_siblings": independent,
        "width_half": means[:, None] + 0.5 * (correct - means[:, None]),
    }
    # The center depends on observations, so the check can respond to ignored X.
    centers = observations @ observation_matrix
    arrays = {
        "truth": truth, "observations": observations, "noise_sigma": sigma,
        "observation_matrix": observation_matrix, "prior_covariance": covariance,
        "posterior_mean": means, "posterior_covariance": covariances,
        "positions": positions, "parent_id": np.arange(parents),
    }
    report = {
        "kind": "analytic_diagnostic_fixture_not_cosmological_data",
        "seed": seed, "parents": parents, "draws": draws,
        "mira_mass_draws": draws - 1,
        "mira_finite_sample_null_mean": (2 * (draws - 1) + 3) / (3 * (draws + 1)),
        "mira_activated": False, "power_study_performed": False,
        "candidates": {},
    }
    for name, sample in candidates.items():
        arrays[f"draws_{name}"] = sample
        mira = mira_per_parent(truth, sample, centers)
        # A data-dependent test quantity is necessary to detect a prior-only model.
        observed_error = (sample @ observation_matrix.T - observations[:, None]) / sigma[:, None, None]
        truth_error = (truth @ observation_matrix.T - observations) / sigma[:, None]
        discrepancy_rank = np.sum(np.sum(observed_error**2, axis=-1) < np.sum(truth_error**2, axis=-1)[:, None], axis=1)
        sum_rank = np.sum(sample.sum(axis=-1) < truth.sum(axis=-1)[:, None], axis=1)
        arrays[f"mira_{name}"] = mira
        arrays[f"data_discrepancy_rank_{name}"] = discrepancy_rank
        arrays[f"sum_rank_{name}"] = sum_rank
        report["candidates"][name] = {
            "mira_mean": float(mira.mean()),
            "mira_parent_standard_error": float(mira.std(ddof=1) / np.sqrt(parents)),
            "sum_rank_mean_fraction": float(sum_rank.mean() / draws),
            "data_discrepancy_rank_mean_fraction": float(discrepancy_rank.mean() / draws),
            "mean_posterior_sum_variance": float(np.var(sample.sum(axis=-1), axis=1, ddof=1).mean()),
        }
    report["analytic_mean_sum_variance"] = float(covariances.sum(axis=(1, 2)).mean())
    report["analytic_independent_sibling_sum_variance"] = float(broken_cov.sum(axis=(1, 2)).mean())
    report["limitations"] = [
        "one small synthetic panel does not establish diagnostic power",
        "no realistic survey masks or nonlinear tides used",
        "rank means alone do not test rank uniformity",
        "MIRA is supplementary until its full activation study passes",
    ]
    return arrays, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    allowed = (REPO / "docs/evidence/e2e_field_v1").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("use a new child of the E2E evidence namespace")
    arrays, report = make_fixture()
    output.mkdir(parents=True, exist_ok=False)
    target = output / "analytic_field_fixture.npz"
    with target.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["fixture_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["fixture_bytes"] = target.stat().st_size
    with (output / "ANALYTIC_FIXTURE.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
