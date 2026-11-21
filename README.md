# ArtificialCanineEngine (ACE)

A simulated golden retriever puppy whose personality is shaped by how it is raised.

Two puppies start from the same brain. One owner is patient and rewards calm behaviour; another is inconsistent. Over a simulated season they should grow into measurably different dogs. The brain behind that is built from parts that each come from real-dog data or real neuroscience, and every decision it makes can be inspected. Everything runs offline on a laptop, with a small embedded board (Jetson Orin Nano) as the later target.

This README walks through what is in the repository today, how the pieces fit, and where it is going.

---

## The brain, in one picture

```
              what the dog perceives
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
   Beliefs        Learning center      Emotions + needs
   "can this      "do I like this,     fear, play, seeking,
    hurt me?"      what pays off?"     frustration, panic, mood
        │              │                  │
        └──────► Scorer ◄─────────────────┘
                   ▲   "which option fits this moment best?"
                   │
              Personality
       (trainability, excitability, attachment,
        energy, boldness, impulsivity)
                   │
                   ▼
        Small local language model
     picks from the scorer's shortlist,
       writes a one-line dog "thought"
                   │
                   ▼
               behaviour
```

Each box is a separate module with its own tests. They are built and checked one at a time, before anything is wired around them.

---

## What is here today

| Part | File | What it does |
|---|---|---|
| Personality prior | `data/derived/personality_prior.json` | Means, spreads and correlations of six traits for golden retriever puppies, from Darwin's Ark (thousands of owner-surveyed dogs) |
| Emotion prior | `data/derived/emotion_prior.json` | Seven emotional tendencies for golden puppies, their correlation with the traits, and how fast each emotion fades |
| Behaviour targets | `data/derived/vidopet_targets.json` | How strongly each trait predicts filmed behaviour in a standard dog personality test (VIDOPET), with bootstrap intervals |
| Personality | `src/ace/core/personality.py` | Draws a puppy's starting traits from the prior |
| Beliefs | `src/ace/core/beliefs.py` | One belief per thing the dog has met: how likely it is to hurt, and how sure the dog is |
| Measuring behaviour | `src/ace/eval/sequences.py` | Turns a run of the dog's choices into phases, timings and intensities, so behaviour can be tested |
| Shared types | `src/ace/contracts.py` | The small frozen dataclasses passed between modules |

The raw datasets are not included, only summary statistics derived from them. See `NOTICE`.

---

## Walkthrough

### Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Python 3.12 or newer. The brain itself needs only numpy.

### 1. Every puppy is different

A puppy's six traits are drawn together from the golden retriever prior, so they correlate the way they do in real dogs. The same seed always gives the same puppy.

```python
from pathlib import Path
import numpy as np
from ace.core.personality import load_prior, sample_puppy

prior = load_prior(Path("data/derived/personality_prior.json"))
rng = np.random.default_rng(7)
print(sample_puppy(rng, prior))
```

```
# rounded
Traits(trainability=0.67, excitability=0.46, attachment=0.34, energy=0.40, boldness=0.71, impulsivity=0.60)
```

Traits are percentiles among all surveyed dogs (0 to 1). Later, upbringing moves them.

### 2. Learning what is dangerous

The dog keeps a belief about each thing it has met: two counts, bad things that happened near it (α) and quiet time spent with it (β). This is a Beta–Bernoulli belief:

- `harm = α / (α + β)`: how dangerous the thing seems.
- `unsure`: how spread out the belief still is, from 1 (no idea) towards 0 (very sure).

That one small model gives several real dog behaviours:

- **Fear:** a scary event raises harm.
- **Habituation:** calm exposure lowers it, faster up close.
- **Spontaneous recovery:** sleep fades half of what was learned, so some fear comes back the next day.
- **Generalisation:** a new thing inherits part of what the dog learned about its category.

```python
from ace.core.beliefs import Beliefs

dog = Beliefs()
dog.meet("vacuum_1", "vacuum", familiarity=0.4)
dog.on_innate("vacuum_1", "vacuum", b=-0.8)            # it roars
for _ in range(40):                                    # 40 quiet visits, about 1 m away
    dog.observe("vacuum_1", "vacuum", close=0.8, dt_s=2.0)
dog.sleep()
dog.meet("vacuum_2", "vacuum", familiarity=0.4)        # a different vacuum
```

| Moment | harm |
|---|---|
| first sight of vacuum_1 | 0.28 |
| after it roars | 0.57 |
| after 40 calm visits | 0.24 |
| next morning | 0.25 |
| first sight of vacuum_2 | 0.26 |

The second vacuum starts slightly *less* feared than the first did, because overall the dog's time with vacuums was mostly calm. `unsure` feeds the decision to go and check something out: checking is worth most when the dog doesn't yet know.

### 3. Testing behaviour without one right answer

There is no single correct way for a dog to react to a scare, but there are generally right and generally wrong ones. So behaviour is tested as **sequences**: a scared dog backs away (A), may check the thing out (C), returns to its toy (R), and eventually plays (P). `measure_run` labels each moment of a run with its phase and measures it, and the tests judge those measurements.

```python
from ace.eval.sequences import SCARE_AND_RECOVER, measure_run

timeline = ([(t, "retreat_from:vacuum_1") for t in (0, 2)]
          + [(t, "investigate:vacuum_1") for t in (4, 6)]
          + [(t, "play_with:ball_1") for t in range(8, 22, 2)])
m = measure_run(SCARE_AND_RECOVER, timeline,
                roles={"trigger": "vacuum_1", "toy": "ball_1"})
```

```
phases         A A C C R R R R R P P
first_reached  {'A': 0, 'C': 4, 'R': 8, 'P': 18}
in_order       True
time_in        {'A': 4, 'C': 4, 'R': 10, 'P': 4}
```

It also reports backsteps (falling back to an earlier phase), how many times the dog switched goals, and whether the intensity of play tracks the emotion driving it. Sequences are data, so a new situation is a new definition, not new code.

---

## Design choices

- **Grounded in real dogs.** Starting personalities and emotions come from large owner surveys of real golden retrievers. Behaviour targets come from filmed personality tests. Numbers that are still guesses are labelled as guesses.
- **Explainable.** Every choice comes with the terms that produced it. No module is a black box that "just learned it".
- **Decide first, then learn.** The decision-making core is built and tested on hand-made moments before the learning center, and both before the simulated world.
- **Deterministic.** Every random draw takes a generator passed in, so any run replays exactly from its seed.
- **Small and offline.** No cloud services. The language model is small enough to run locally, and only ever picks from options the scorer offers.

---

## Roadmap

| Step | | Status |
|---|---|---|
| 1 | Personality prior from Darwin's Ark | done |
| 2 | Behaviour targets from VIDOPET | done |
| 3 | Emotion prior from Darwin's Ark and C-BARQ | done |
| 4 | Decision core: emotions, beliefs, fit scorer, selection, language-model reviewer | in progress |
| 5 | Learning center modelled on the fruit fly mushroom body, compared against simpler learners on a test bench, including a spiking version | next |
| 6 | Smallest world loop: a room, perception, skills, one scripted owner, exact replay | |
| 7 | Brain cam: a live view of every decision | |
| 8 | Personality drift with upbringing | |
| 9 | Memory and sleep consolidation | |
| 10 | Hearing: learning words from the owner's voice | |
| 11 | Scripted owners and the main experiment: same puppy, different upbringing, different dog | |

---

## Data and credits

ACE uses summary statistics derived from Darwin's Ark (Morrill et al. 2022, CC0), the VIDOPET database (Turcsán et al. 2018, CC BY 4.0) and C-BARQ survey data (Ghirlanda et al. 2013, CC BY 4.0). Full citations and the changes made are in [`NOTICE`](NOTICE).
