from dataclasses import astuple, fields
from pathlib import Path

import numpy as np

from ace.contracts import Traits
from ace.core.personality import TRAIT_MAX, TRAIT_MIN, TRAIT_ORDER, load_prior, sample_puppy

PRIOR = load_prior(Path(__file__).parent.parent / "data/derived/personality_prior.json")


def sample_many(n: int, seed: int = 0) -> np.ndarray:
    """n puppies from one generator, as an (n, 6) array in TRAIT_ORDER."""
    rng = np.random.default_rng(seed)
    return np.array([astuple(sample_puppy(rng, PRIOR)) for _ in range(n)])


def test_trait_order_matches_traits_fields():
    # If these disagree, energy silently becomes boldness and nothing crashes.
    assert TRAIT_ORDER == tuple(f.name for f in fields(Traits))


def test_prior_shapes():
    assert PRIOR.mean.shape == (6,)
    assert PRIOR.sd.shape == (6,)
    assert PRIOR.corr.shape == (6, 6)
    assert np.allclose(np.diag(PRIOR.corr), 1.0)


def test_same_seed_same_puppy():
    a = sample_puppy(np.random.default_rng(7), PRIOR)
    b = sample_puppy(np.random.default_rng(7), PRIOR)
    assert a == b


def test_different_seeds_differ():
    a = sample_puppy(np.random.default_rng(1), PRIOR)
    b = sample_puppy(np.random.default_rng(2), PRIOR)
    assert a != b


def test_traits_in_bounds():
    puppies = sample_many(1_000)
    assert puppies.min() >= TRAIT_MIN
    assert puppies.max() <= TRAIT_MAX


def test_population_matches_prior_mean():
    # Clipping shifts each mean by at most ~0.013 (measured), so 0.02 is enough.
    puppies = sample_many(10_000)
    assert np.allclose(puppies.mean(axis=0), PRIOR.mean, atol=0.02)


def test_population_keeps_correlations():
    # Clipping weakens correlations slightly (~0.01 measured), so 0.03 is enough.
    corr = np.corrcoef(sample_many(10_000), rowvar=False)
    exc, imp, train = (TRAIT_ORDER.index(t) for t in ("excitability", "impulsivity", "trainability"))
    assert abs(corr[exc, imp] - PRIOR.corr[exc, imp]) < 0.03
    assert abs(corr[train, imp] - PRIOR.corr[train, imp]) < 0.03
