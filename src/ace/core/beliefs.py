from ace.contracts import ThingBelief
import math

FLAT_SD = math.sqrt(1/12)
STARTLE_WEIGHT = 3.0

def beta_mean(alpha, beta) -> float:
    return alpha / (alpha + beta)

def beta_sd(alpha, beta) -> float:
    return math.sqrt((alpha * beta)/((alpha + beta) ** 2 * (alpha + beta + 1)))

def unsure(alpha, beta) -> float:
    if alpha ==  1 and beta == 1:
        return 1.0 
    return beta_sd(alpha=alpha, beta=beta) / FLAT_SD

class Beliefs:
    def __init__(self) -> None:
        self.known = {}
    
    def meet(self, entity_id, category, familiarity) -> None:
        if 0.0 > familiarity or familiarity > 1.0:
            raise ValueError("[meet]: familiarity is out of bounds")
        if entity_id not in self.known:
            alpha_0 = 1
            beta_0 = 1 + 4 * familiarity
            self.known[entity_id] = { "category" : category, "alpha_0": alpha_0, "beta_0": beta_0, "alpha": alpha_0, "beta": beta_0 } 
    
    def get(self, entity_id) -> ThingBelief:
        if entity_id not in self.known:
            raise KeyError("[get]: entity_id not found")
        thing = self.known[entity_id]
        alpha = thing["alpha"]       
        beta = thing["beta"]
        return ThingBelief(harm=beta_mean(alpha, beta), unsure=unsure(alpha, beta), alpha=alpha, beta=beta)
    
    def on_innate(self, entity_id, category, b) -> None:
        self.meet(entity_id, category, 0.0)
        if b < 0:
            self.known[entity_id]["alpha"] += abs(b) * STARTLE_WEIGHT
    
    def on_punished(self, entity_id, category, r) -> None:
        self.meet(entity_id, category, 0.0)
        if r < 0:
            self.known[entity_id]["alpha"] += abs(r) * 2.0
            
    def observe(self, entity_id, category, close, dt_s) -> None:
        if close < 0.0 or close > 1.0:
            raise ValueError("[observe]: close is out of bounds")
        if dt_s < 0:
            raise ValueError("[observe]: dt_s is less than 0")
        self.meet(entity_id, category, 0.0)
        self.known[entity_id]["beta"] += 0.25 * max(close, 0.3) * dt_s / 2
