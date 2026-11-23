from dataclasses import replace, dataclass
from ace.contracts import Emotions, EmotionalTendencies, Needs

FAST = ("seeking", "fear", "frustration", "panic", "play")

@dataclass(frozen=True, slots=True)
class Dials:
    happiness: float
    agitation: float
    activation: float

def dials(e: Emotions, weights: dict[str, dict[str, float]]) -> Dials:
    def weighted(dial):
        return sum(weights[dial][k] * x[k] for k in x)
    x = {}
    for name in FAST:
        x[name] = getattr(e, name) - 0.1
    x["mood"] = e.mood / 2
    happiness = clip(2 * weighted("happiness"), -1, 1)
    agitation = clip(2 * weighted("agitation"), 0, 1)
    activation = max(getattr(e, name) for name in FAST)
    return Dials(happiness=happiness, agitation=agitation, activation=activation)
            
def clip(x, lo, hi):
    return max(lo, min(hi, x))

def gain(t) -> float:
    return 0.5 + t

def decay_factor(dt_s, half_life_s) -> float:
    if dt_s < 0 or half_life_s <= 0:
        raise ValueError("[decay_factor]: dt_s < 0 or half_life_s <= 0")
    return 0.5 ** (dt_s / half_life_s)

def relax(x, rest, dt_s, half_life_s) -> float:
    return rest + (x - rest) * decay_factor(dt_s, half_life_s)

def calm_multiplier(p_calm, bands) -> float:
    for up_to_level, multiplier in bands:
        if p_calm <= up_to_level:
            return multiplier
    return bands[-1][1]

def most_harmful(things) -> tuple[float, float]:
    if len(things) == 0:
        return (0.0, 0.0)
    return max(things)

class EmotionCore:
    def __init__(self, p: EmotionalTendencies, calm_bands) -> None:
        self.p = p
        m = calm_multiplier(p.calm_down, calm_bands)
        self.fast_half_life_s = 111.1 * m
        self.fear_half_life_s = self.fast_half_life_s * (0.5 + p.fear)
        self.mood_half_life_s = 833 * m
        self.wake()
        
    def mood_rest(self) -> float:
        return 0.8 * (self.p.mood - 0.5)
    
    def wake(self) -> Emotions:
        p = self.p
        mood = self.mood_rest()
        good_vibes = 0.1 * max(0, mood)
        bad_vibes = 0.1 * max(0, -mood)
        self._state = Emotions(
            seeking=0.2*p.seeking + good_vibes,
            play=0.2*p.play + good_vibes,
            panic=0.2*p.panic + bad_vibes,
            fear=0.2*p.fear + bad_vibes,
            frustration=0.2*p.frustration + bad_vibes,
            mood=mood
        )
        return self._state
    
    @property
    def state(self) -> Emotions:
        return self._state
    
    def step(self, dt_s, needs, owner_absent=False, playing=False, harm=0.0, close=0.0, interest=0.0) -> Emotions:
        rest = self.rest_levels(needs, owner_absent, playing, harm, close, interest)
        e = self._state
        seeking = relax(e.seeking, rest["seeking"], dt_s, self.fast_half_life_s)
        fear = relax(e.fear, rest["fear"], dt_s, self.fear_half_life_s)
        play = relax(e.play, rest["play"], dt_s, self.fast_half_life_s)
        panic = relax(e.panic, rest["panic"], dt_s, self.fast_half_life_s)
        frustration = relax(e.frustration, rest["frustration"], dt_s, self.fast_half_life_s)
        target = self.mood_rest() + 0.2 * (seeking + play) - 0.6 * max(fear, frustration, panic) - 0.8 * max(0, max(needs.play, needs.social, needs.novelty) - 0.5)
        mood = relax(e.mood, target, dt_s, self.mood_half_life_s)
        self._state = Emotions(
            seeking=clip(seeking, 0, 1),
            play=clip(play, 0, 1),
            panic=clip(panic, 0, 1),
            fear=clip(fear, 0, 1),
            frustration=clip(frustration, 0, 1),
            mood=clip(mood, -1, 1)
        )
        return self._state
            
    def rest_levels(self, needs: Needs, owner_absent=False, playing=False, harm=0.0, close=0.0, interest=0.0) -> dict[str, float]:
        for name, x in (("harm", harm), ("close", close), ("interest", interest)):
            if not 0 <= x <= 1:
                raise ValueError(f"[rest_levels]: {name} must be in [0, 1], got {x}")
        p = self.p
        mood = self._state.mood
        good_vibes = 0.1 * max(0, mood)
        bad_vibes = 0.1 * max(0, -mood)
        rest = {
            "seeking": 0.2 * p.seeking + good_vibes,
            "play": 0.2 * p.play + good_vibes,
            "panic": 0.2 * p.panic + bad_vibes,
            "fear": 0.2 * p.fear + bad_vibes,
            "frustration": 0.2 * p.frustration + bad_vibes,
        }
        if owner_absent:
            rest["panic"] += 0.4 * needs.social * gain(p.panic)
        if playing:
            rest["play"] += 0.5
        rest["fear"] += 0.5 * harm * close ** 2
        rest["seeking"] += 0.3 * interest * gain(p.seeking)
        return rest
    
    def on_innate(self, b) -> Emotions:
        if not -1 <= b <= 1:
            raise ValueError(f"[on_innate]: b must be in [-1, 1], got {b}")
        if b >= 0:
            return self._state
        e = self._state
        self._state = replace(e,
            fear=clip(e.fear + 0.8 * abs(b), 0, 1),
            mood=clip(e.mood + 0.2 * b, -1, 1),
        )
        return self._state
    
    def on_reward(self, r, playing=False) -> Emotions:
        e = self._state
        if r > 0:
            if playing:
                self._state = replace(e,
                    play=clip(e.play + 0.3 * r * gain(self.p.play), 0, 1),
                    mood=clip(e.mood + 0.2 * r, -1, 1),
                )
            else:
                self._state = replace(e,
                    seeking=clip(e.seeking + 0.3 * r * gain(self.p.seeking), 0, 1),
                    mood=clip(e.mood + 0.2 * r, -1, 1),
                )
        if r < 0:
            self._state = replace(e,
                    fear=clip(e.fear + 0.4 * abs(r) * gain(self.p.fear), 0, 1),
                    mood=clip(e.mood + 0.2 * r, -1, 1),
                )
        return self._state
    
    def on_blocked(self) -> Emotions:
        e = self._state
        self._state = replace(e,
            frustration=clip(e.frustration + 0.4 * gain(self.p.frustration), 0, 1)
        )
        return self._state
    
    def on_novel(self) -> Emotions:
        e = self._state
        self._state = replace(e,
            seeking=clip(e.seeking + 0.2 * gain(self.p.seeking), 0, 1)
        )
        return self._state

    def on_owner_left(self) -> Emotions:
        e = self._state
        self._state = replace(e,
            panic=clip(e.panic + 0.3 * gain(self.p.panic), 0, 1)
        )
        return self._state
    
    def on_owner_returned(self) -> Emotions:
        p = self.p
        e = self._state
        rest = 0.2 * p.panic + 0.1 * max(0, -e.mood)
        self._state = replace(e,
            panic=clip(rest + 0.5 * (e.panic - rest), 0, 1),
            play=clip(e.play + 0.3 * gain(p.play), 0, 1),
            seeking=clip(e.seeking + 0.2 * gain(p.seeking), 0, 1),
            mood=clip(e.mood + 0.1, -1, 1),
        )
        return self._state
    
    def on_tone(self, tone) -> Emotions:
        p = self.p
        e = self._state
        self._state = replace(e,
                              
            fear=clip(e.fear + 0.2 * tone.stern * gain(p.fear), 0, 1),
            mood=clip(e.mood + 0.1 * tone.valence, -1, 1),
        )
        return self._state