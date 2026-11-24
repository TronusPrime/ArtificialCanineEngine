from dataclasses import replace, dataclass
from ace.contracts import Emotions, EmotionalTendencies, Needs

FAST = ("seeking", "fear", "frustration", "panic", "play")

# Speeds
FAST_HALF_LIFE_S = 111.1
MOOD_HALF_LIFE_S = 833.0

# Resting levels
REST_PER_TENDENCY = 0.2
MOOD_INTO_REST = 0.1
PANIC_ALONE = 0.4
PLAY_WHILE_PLAYING = 0.5
FEAR_NEAR_HARM = 0.5
SEEKING_FROM_INTEREST = 0.3

# Mood
MOOD_REST_SCALE = 0.8
MOOD_FROM_GOOD = 0.2
MOOD_FROM_BAD = 0.6
MOOD_FROM_NEEDS = 0.8
NEED_THRESHOLD = 0.5

# Events
STARTLE_FEAR = 0.8
STARTLE_MOOD = 0.2
REWARD_BOOST = 0.3
REWARD_MOOD = 0.2
PUNISH_FEAR = 0.4
BLOCKED_FRUSTRATION = 0.4
NOVEL_SEEKING = 0.2
OWNER_LEFT_PANIC = 0.3
RETURN_PANIC_KEEP = 0.5
RETURN_PLAY = 0.3
RETURN_SEEKING = 0.2
RETURN_MOOD = 0.1
TONE_MOOD = 0.1
TONE_STERN_FEAR = 0.2

# Dials
DIAL_REST = 0.1
DIAL_MOOD_SCALE = 0.5
DIAL_SCALE = 2.0

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
        x[name] = getattr(e, name) - DIAL_REST
    x["mood"] = e.mood * DIAL_MOOD_SCALE
    happiness = clip(DIAL_SCALE * weighted("happiness"), -1, 1)
    agitation = clip(DIAL_SCALE * weighted("agitation"), 0, 1)
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
        self.fast_half_life_s = FAST_HALF_LIFE_S * m
        self.fear_half_life_s = self.fast_half_life_s * gain(p.fear)
        self.mood_half_life_s = MOOD_HALF_LIFE_S * m
        self.wake()
        
    def mood_rest(self) -> float:
        return MOOD_REST_SCALE * (self.p.mood - 0.5)
    
    def _base_rest(self, mood: float) -> dict[str, float]:
        p = self.p
        good_vibes = MOOD_INTO_REST * max(0, mood)
        bad_vibes = MOOD_INTO_REST * max(0, -mood)
        return {
            "seeking": REST_PER_TENDENCY * p.seeking + good_vibes,
            "fear": REST_PER_TENDENCY * p.fear + bad_vibes,
            "frustration": REST_PER_TENDENCY * p.frustration + bad_vibes,
            "panic": REST_PER_TENDENCY * p.panic + bad_vibes,
            "play": REST_PER_TENDENCY * p.play + good_vibes,
        }
        
    def wake(self) -> Emotions:
        mood = self.mood_rest()
        emotes = self._base_rest(mood)
        self._state = Emotions(**emotes, mood=mood)
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
        target = (self.mood_rest() + MOOD_FROM_GOOD * (seeking + play)
                  - MOOD_FROM_BAD * max(fear, frustration, panic)
                  - MOOD_FROM_NEEDS * max(0, max(needs.play, needs.social, needs.novelty) - NEED_THRESHOLD))
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
        rest = self._base_rest(mood)
        if owner_absent:
            rest["panic"] += PANIC_ALONE * needs.social * gain(p.panic)
        if playing:
            rest["play"] += PLAY_WHILE_PLAYING
        rest["fear"] += FEAR_NEAR_HARM * harm * close ** 2
        rest["seeking"] += SEEKING_FROM_INTEREST * interest * gain(p.seeking)
        return rest
    
    def on_innate(self, b) -> Emotions:
        if not -1 <= b <= 1:
            raise ValueError(f"[on_innate]: b must be in [-1, 1], got {b}")
        if b >= 0:
            return self._state
        e = self._state
        self._state = replace(e,
            fear=clip(e.fear + STARTLE_FEAR * abs(b), 0, 1),
            mood=clip(e.mood + STARTLE_MOOD * b, -1, 1),
        )
        return self._state
    
    def on_reward(self, r, playing=False) -> Emotions:
        e = self._state
        if r > 0:
            if playing:
                self._state = replace(e,
                    play=clip(e.play + REWARD_BOOST * r * gain(self.p.play), 0, 1),
                    mood=clip(e.mood + REWARD_MOOD * r, -1, 1),
                )
            else:
                self._state = replace(e,
                    seeking=clip(e.seeking + REWARD_BOOST * r * gain(self.p.seeking), 0, 1),
                    mood=clip(e.mood + REWARD_MOOD * r, -1, 1),
                )
        if r < 0:
            self._state = replace(e,
                    fear=clip(e.fear + PUNISH_FEAR * abs(r) * gain(self.p.fear), 0, 1),
                    mood=clip(e.mood + REWARD_MOOD * r, -1, 1),
                )
        return self._state
    
    def on_blocked(self) -> Emotions:
        e = self._state
        self._state = replace(e,
            frustration=clip(e.frustration + BLOCKED_FRUSTRATION * gain(self.p.frustration), 0, 1)
        )
        return self._state
    
    def on_novel(self) -> Emotions:
        e = self._state
        self._state = replace(e,
            seeking=clip(e.seeking + NOVEL_SEEKING * gain(self.p.seeking), 0, 1)
        )
        return self._state

    def on_owner_left(self) -> Emotions:
        e = self._state
        self._state = replace(e,
            panic=clip(e.panic + OWNER_LEFT_PANIC * gain(self.p.panic), 0, 1)
        )
        return self._state
    
    def on_owner_returned(self) -> Emotions:
        p = self.p
        e = self._state
        rest = self._base_rest(e.mood)
        self._state = replace(e,
            panic=clip(rest["panic"] + RETURN_PANIC_KEEP * (e.panic - rest["panic"]), 0, 1),
            play=clip(e.play + RETURN_PLAY * gain(p.play), 0, 1),
            seeking=clip(e.seeking + RETURN_SEEKING * gain(p.seeking), 0, 1),
            mood=clip(e.mood + RETURN_MOOD, -1, 1),
        )
        return self._state
    
    def on_tone(self, tone) -> Emotions:
        p = self.p
        e = self._state
        self._state = replace(e,
            fear=clip(e.fear + TONE_STERN_FEAR * tone.stern * gain(p.fear), 0, 1),
            mood=clip(e.mood + TONE_MOOD * tone.valence, -1, 1),
        )
        return self._state