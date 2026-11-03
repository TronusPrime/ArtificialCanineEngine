import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ace.contracts import Traits

# Fixed order. Every array below follows it.
TRAIT_ORDER = ("trainability", "excitability", "attachment", "energy", "boldness", "impulsivity")

TRAIT_MIN, TRAIT_MAX = 0.05, 0.95


@dataclass(frozen=True)
class Prior:
    mean: np.ndarray   # shape (6,)
    sd: np.ndarray     # shape (6,)
    corr: np.ndarray   # shape (6, 6)


def load_prior(path: Path) -> Prior:
    """Read personality_prior.json into arrays in TRAIT_ORDER."""
    data = json.loads(path.read_text())
    mean = []
    sd = []
    corr = []
    for trait in TRAIT_ORDER:
        mean.append(data["mean"][trait])
        sd.append(data["sd"][trait])
        
        row = []
        for other in TRAIT_ORDER:
            row.append(data["corr"][trait][other])
        corr.append(row)
    return Prior(mean = np.array(mean), sd = np.array(sd), corr = np.array(corr))


# Clip, don't redraw: ~6-12% of values per trait land on a limit.
def sample_puppy(rng: np.random.Generator, prior: Prior) -> Traits:
    """Draw one puppy's starting traits. Same rng state -> same puppy."""
    n = len(prior.sd)                     # 6
    cov = np.zeros((n, n))                # empty 6x6 table
    for i in range(n):
        for j in range(n):
            cov[i, j] = prior.sd[i] * prior.sd[j] * prior.corr[i, j]
    values = rng.multivariate_normal(prior.mean, cov)
    values = np.clip(values, TRAIT_MIN, TRAIT_MAX)
    return Traits(*values.tolist())