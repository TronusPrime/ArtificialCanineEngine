from ace.eval.sequences import SCARE_AND_RECOVER, classify, measure_run, first_reached, follows_emotion, SequenceDef, Phase
import pytest

ROLES = {"trigger": "vacuum_1", "toy": "toy_3"}

def test_no_retreat_means_no_reaction_latency():
    timeline = [(2, "investigate:toy_3"), (4, "chew:toy_3"), (6, "play_with:toy_3"), (8, "play_with:toy_3")]
    m = measure_run(SCARE_AND_RECOVER, timeline, ROLES)
    assert m["reaction_latency"] is None
    
def test_first_reached_mismatched_lengths():
    with pytest.raises(ValueError):
        first_reached(["A", "C"], [2.0])
    
def test_follows_emotion_mismatch():
    timeline = [(2,"retreat_from:vacuum_1"), (4, "investigate:toy_3"), (6, "play_with:toy_3")]
    phases = classify(SCARE_AND_RECOVER, timeline, ROLES)
    with pytest.raises(ValueError):
        follows_emotion(SCARE_AND_RECOVER, timeline, phases, ROLES, [0.1])
    
def test_one_phase_fail():
    test_seq = SequenceDef(
        name="one",
        phases=(
            Phase("P", ("chew:toy",)),
            Phase("P", ("play_with:toy",)),
        ),
        intensity= { "chew:toy": 2, "play_with:toy": 3 },
        builds_up_with="play",
        final_min_intensity=3
    )
    timeline = [(2, "chew:toy_3"), (4, "play_with:toy_3")]
    phases = classify(test_seq, timeline, ROLES)
    assert follows_emotion(test_seq, timeline, phases, ROLES, [0.1, 0.3]) == (None, 0)
    
def test_my_hand_labels_match():
    timeline = [(2, "go_to_owner"), (4, "go_to_owner"), (6, "retreat_from:vacuum_1"), (8, "look_to_owner"), (10, "investigate:vacuum_1"), (12, "go_to_owner"),
                (14, "investigate:toy_3"), (16, "chew:toy_3"), (18, "rest"), (28, "chew:toy_3"), (30, "play_with:toy_3"), (32, "look_to_owner")]
    phases = classify(SCARE_AND_RECOVER, timeline, ROLES)
    assert phases == ["A", "A", "A", "C", "C", None, "R", "R", None, "P", "P", None]
    times = [t for t,_ in timeline]
    assert first_reached(phases, times) == {"A": 2.0, "C": 8.0, "R": 14.0, "P": 28.0}
    

def test_empty_timeline():
    with pytest.raises(ValueError):
        measure_run(SCARE_AND_RECOVER, [], ROLES)
        

@pytest.mark.parametrize(
    "timeline",
    [
        [(2, "retreat_from:vacuum_1"), (2, "go_to_owner"), (4, "go_to_owner")],
        [(4, "retreat_from:vacuum_1"), (2, "go_to_owner"), (6, "go_to_owner")],
    ],
)
def test_bad_samples(timeline):
    with pytest.raises(ValueError):
        measure_run(SCARE_AND_RECOVER, timeline, ROLES)