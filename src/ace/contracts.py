# Traits type

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Traits:
    trainability: float
    excitability: float
    attachment: float
    energy: float
    boldness: float
    impulsivity: float
