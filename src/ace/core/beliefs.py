from ace.contracts import ThingBelief
import math
import copy

FLAT_SD = math.sqrt(1/12)
STARTLE_WEIGHT = 3.0
PUNISH_WEIGHT = 2.0
FAMILIARITY_WEIGHT = 4.0
QUIET_WEIGHT = 0.25
QUIET_STEP_S = 2.0
MIN_CLOSE = 0.3
SLEEP_KEEP = 0.5
CATEGORY_WEIGHT = 0.5

def beta_mean(alpha, beta) -> float:
    return alpha / (alpha + beta)

def beta_sd(alpha, beta) -> float:
    return math.sqrt((alpha * beta)/((alpha + beta) ** 2 * (alpha + beta + 1)))

def unsure(alpha, beta) -> float:
    return beta_sd(alpha=alpha, beta=beta) / FLAT_SD

class Beliefs:
    def __init__(self) -> None:
        self.known = {}
    
    def meet(self, entity_id, category, familiarity) -> None:
        if 0.0 > familiarity or familiarity > 1.0:
            raise ValueError("[meet]: familiarity is out of bounds")
        if entity_id in self.known and self.known[entity_id]["category"] != category:
            raise ValueError("[meet]: category mismatch")
        if entity_id not in self.known:
            bad, quiet = self.category_evidence(category)
            alpha_0 = 1.0 + CATEGORY_WEIGHT * bad
            beta_0 = 1.0 + FAMILIARITY_WEIGHT * familiarity + CATEGORY_WEIGHT * quiet
            self.known[entity_id] = { "category" : category, "alpha_0": alpha_0, "beta_0": beta_0, "alpha": alpha_0, "beta": beta_0 } 
    def get(self, entity_id) -> ThingBelief:
        if entity_id not in self.known:
            raise KeyError("[get]: entity_id not found")
        thing = self.known[entity_id]
        alpha = thing["alpha"]       
        beta = thing["beta"]
        return ThingBelief(harm=beta_mean(alpha, beta), unsure=unsure(alpha, beta), alpha=alpha, beta=beta)
    
    def on_innate(self, entity_id, category, b) -> None:
        if b < -1.0 or b > 1.0:
            raise ValueError("[on_innate]: b is out of bounds")
        self.meet(entity_id, category, 0.0)
        if b < 0:
            self.known[entity_id]["alpha"] += abs(b) * STARTLE_WEIGHT
    
    def on_punished(self, entity_id, category, r) -> None:
        if r < -1.0 or r > 1.0:
            raise ValueError("[on_punished]: r is out of bounds")
        self.meet(entity_id, category, 0.0)
        if r < 0:
            self.known[entity_id]["alpha"] += abs(r) * PUNISH_WEIGHT
            
    def observe(self, entity_id, category, close, dt_s) -> None:
        if close < 0.0 or close > 1.0:
            raise ValueError("[observe]: close is out of bounds")
        if dt_s < 0:
            raise ValueError("[observe]: dt_s is less than 0")
        self.meet(entity_id, category, 0.0)
        self.known[entity_id]["beta"] += QUIET_WEIGHT * max(close, MIN_CLOSE) * dt_s / QUIET_STEP_S
        
    def category_evidence(self, category) -> tuple[float, float]:
        bad, quiet = 0.0, 0.0
        
        for thing in self.known.values():
            if thing["category"] == category:
                bad += thing["alpha"] - thing["alpha_0"]
                quiet += thing["beta"] - thing["beta_0"]
        return (bad, quiet)
    
    def sleep(self) -> None:
        for thing in self.known.values():
            thing["alpha"] = thing["alpha_0"] + (thing["alpha"] - thing["alpha_0"]) * SLEEP_KEEP
            thing["beta"]  = thing["beta_0"] + (thing["beta"] - thing["beta_0"]) * SLEEP_KEEP
            
    def state_dict(self) -> dict:
        return copy.deepcopy(self.known)
    
    def load_state_dict(self, state) -> None:
        self.known = {eid: dict(thing) for eid, thing in state.items()}