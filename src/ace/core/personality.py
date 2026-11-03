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
    # 1. json.loads(path.read_text())
    # 2. mean: one value per trait name, in TRAIT_ORDER
    # 3. sd:   same
    # 4. corr: 6x6, row = trait a, column = trait b, both looped in TRAIT_ORDER
    #    (the JSON stores it as data["corr"][a][b])
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


def sample_puppy(rng: np.random.Generator, prior: Prior) -> Traits:
    """Draw one puppy's starting traits. Same rng state -> same puppy."""
    # 1. cov = how much each trait varies, combined with how traits move together
    # 2. draw 6 correlated values around prior.mean
    # 3. clip to [TRAIT_MIN, TRAIT_MAX]  (or redraw, per your 4d decision)
    # 4. Traits(*values): works because Traits fields are in TRAIT_ORDER
    n = len(prior.sd)                     # 6
    cov = np.zeros((n, n))                # empty 6x6 table
    for i in range(n):
        for j in range(n):
            cov[i, j] = prior.sd[i] * prior.sd[j] * prior.corr[i, j]
    values = rng.multivariate_normal(prior.mean, cov)
    values = np.clip(values, TRAIT_MIN, TRAIT_MAX)
    return Traits(*values.tolist())