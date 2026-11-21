from ace.core.beliefs import Beliefs, unsure
import pytest


def test_40_quiet_exposures_bring_harm_below_0_3():
    bel = Beliefs()
    bel.meet(entity_id="vacuum_1", category="vacuum", familiarity=0.4)
    bel.on_innate(entity_id="vacuum_1", category="vacuum", b=-0.8)
    for _ in range(40):
        bel.observe(entity_id="vacuum_1", category="vacuum", close=0.8, dt_s=2.0)
    assert bel.get("vacuum_1").harm < 0.3
    
    
def test_unsure_is_highest_at_an_even_split():
    peak = unsure(5, 5)
    for alpha in range(1, 10):
        if alpha == 5:
            continue
        beta = 10 - alpha
        assert peak > unsure(alpha, beta)
        
def test_on_innate_b_bounds():
    bel = Beliefs()
    bel.meet(entity_id="vacuum_1", category="vacuum", familiarity=0.4)
    with pytest.raises(ValueError):
        bel.on_innate(entity_id="vacuum_1", category="vacuum", b=-5.8)
    with pytest.raises(ValueError):
        bel.on_innate(entity_id="vacuum_1", category="vacuum", b=5.8)
    
def test_on_punished_r_bounds():
    bel = Beliefs()
    bel.meet(entity_id="vacuum_1", category="vacuum", familiarity=0.4)
    with pytest.raises(ValueError):
        bel.on_punished(entity_id="vacuum_1", category="vacuum", r=-5.8)
    with pytest.raises(ValueError):
        bel.on_punished(entity_id="vacuum_1", category="vacuum", r=5.8)