from ace.core.emotions import EmotionCore
from ace.contracts import EmotionalTendencies, Needs, Traits
from ace.runtime.needs import NeedsModel, NeedsInputs
import json
from pathlib import Path

PRIOR_PATH = Path(__file__).parent.parent / "data/derived/emotion_prior.json"
_PRIOR = json.loads(PRIOR_PATH.read_text())
BANDS = tuple((b["up_to_level"], b["multiplier"]) for b in _PRIOR["speeds"]["calm_down_bands"])
AVG = EmotionalTendencies(*[0.5] * 7)
AVG_TRAITS = Traits(*[0.5] * 6)
CALM = Needs(battery=1.0, play=0.0, social=0.0, novelty=0.0)


def test_mood_lingers():
    emotions = EmotionCore(AVG, BANDS)
    mood_b4 = emotions.state.mood
    emotions.on_innate(-0.8)
    emotions.step(600, CALM)
    assert emotions.rest_levels(CALM)["fear"] - 0.05 <= emotions.state.fear <= emotions.rest_levels(CALM)["fear"] + 0.05
    assert emotions.state.mood < mood_b4
    emotions.step(3000, CALM)
    assert mood_b4 - 0.05 <= emotions.state.mood <= mood_b4 + 0.05
    

def test_owner_away_panic_increases():
    emotions = EmotionCore(AVG, BANDS)
    need_model = NeedsModel(AVG_TRAITS)
    need_inputs = NeedsInputs(owner_absent=True)
    need_model.wake()
    for _ in range(0, 3600, 1):
        need_model.step(1, need_inputs)
        emotions.step(1, need_model.state, owner_absent = True)
    assert emotions.state.panic >= 0.4
    assert emotions.state.mood < 0.0