
from dataclasses import dataclass

# Traits- What is the puppy like?
@dataclass(frozen=True, slots=True)
class Traits:
    trainability: float
    excitability: float
    attachment: float
    energy: float
    boldness: float
    impulsivity: float
    
# ThingBelief- beliefs about a thing
@dataclass(frozen=True, slots=True)
class ThingBelief:
    harm: float
    unsure: float
    alpha: float
    beta: float

# Needs- What a puppy needs
@dataclass(frozen=True, slots=True)
class Needs:
    battery: float
    play: float
    social: float
    novelty: float

# Emotions- What a puppy feels now
@dataclass(frozen=True, slots=True)
class Emotions:
    seeking: float
    fear: float
    frustration: float
    panic: float
    play: float
    mood: float

# Emotional Tendencies- how sensitive is a pup to certain emotions
@dataclass(frozen=True, slots=True)
class EmotionalTendencies:
    seeking: float
    fear: float
    frustration: float
    panic: float
    play: float
    mood: float
    calm_down: float

# Tone- how a person said something.
@dataclass(frozen=True, slots=True)
class Tone:
    valence: float
    arousal: float
    stern: float
