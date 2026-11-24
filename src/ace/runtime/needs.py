from dataclasses import dataclass, replace
from ace.contracts import Needs
from ace.core.emotions import clip, gain

@dataclass(frozen = True, slots = True)
class NeedsInputs:
    motion: str = "rest"
    docked: bool = False
    playing: bool = False
    exploring: bool = False
    owner_absent: bool = False
    owner_dist_m: float = 5.0
    owner_interacting: bool = False
    owner_events: int = 0
    unfamiliar_seen: int = 0

class NeedsModel:
    def __init__(self, traits) -> None:
        self.traits = traits
        
    @property
    def state(self) -> Needs:
        return self._state
        
    def wake(self, days_away=0) -> Needs:
        if days_away < 0:
            raise ValueError("[wake]: days_away is negative")
        self._state = Needs(battery=1.0, play=0.5, social=1.0 if days_away > 0 else 0.3, novelty=0.4)
        return self._state
    
    def step(self, dt_s, inp: NeedsInputs) -> Needs:
        if dt_s < 0.0:
            raise ValueError("[step]: dt_s is out of bounds")
        if inp.motion not in ("walk", "rest", "run"):
            raise ValueError("[step]: unknown motion")
        e = self._state
        minutes = dt_s / 60.0
        battery_rate = 0.0
        play_rate = 0.0
        social_rate = 0.0
        novelty_rate = 0.0

        # battery
        if inp.docked:
            battery_rate += 0.3
        else:
            if inp.motion == "rest":
                if inp.playing:
                    battery_rate -= 0.12
                else:
                    battery_rate -= 0.02
            if inp.motion == "walk":
                if inp.playing:
                    battery_rate -= 0.12
                else:
                    battery_rate -= 0.05
            if inp.motion == "run":
                battery_rate -= 0.12
        
        # play
        if not inp.playing:
            play_rate += 0.08 * gain(self.traits.energy)
        else:
            play_rate -= 0.25
            
        # social
        if inp.owner_absent or inp.owner_dist_m > 2.0:
            social_rate += 0.06 * gain(self.traits.attachment)
        elif not inp.owner_interacting:
            social_rate += 0.02  
        if not inp.owner_absent and inp.owner_dist_m < 1.2:
            social_rate -= 0.1
        
        # novelty_rate
        novelty_rate += 0.05 * gain(self.traits.excitability)
        if inp.exploring:
            novelty_rate -= 0.05
            
        self._state = replace(e, 
                        battery=clip(e.battery + (battery_rate * minutes), 0, 1),
                        play=clip(e.play + (play_rate * minutes), 0, 1),
                        social=clip(e.social + (social_rate * minutes - 0.3*inp.owner_events), 0, 1),
                        novelty=clip(e.novelty + novelty_rate * minutes - 0.2 * inp.unfamiliar_seen, 0, 1)
                        )
        return self._state
    