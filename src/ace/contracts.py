
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