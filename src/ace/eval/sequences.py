from dataclasses import dataclass, field
from collections.abc import Mapping
import numpy as np


@dataclass(frozen=True)
class Phase:
    name: str                              # "A", "C", "R", "P", ...
    goals: tuple[str, ...]                 # option ids, with role names as targets: "retreat_from:trigger", "zoomies"
    until_phases: tuple[str, ...] = ()     # row stops matching once any earlier sample was labelled with one of these
    until_intensity: int | None = None     # row stops matching once any earlier sample had at least this intensity
    within_s: float | None = None          # row matches only if an earlier sample labelled with one of `after`
    after: tuple[str, ...] = ()            #   happened at most `within_s` seconds ago (inclusive)

@dataclass(frozen=True)
class SequenceDef:
    name: str
    phases: tuple[Phase, ...]              # rows in order; a sample's phase is the first row that matches
    optional: tuple[str, ...] = ()         # phase names that may be skipped
    intensity: Mapping[str, int] = field(default_factory=dict)   # option id (with roles) -> 1..4
    builds_up_with: str | None = None      # name of the driving emotion, if intensity should rise with it
    builds_down_with: str | None = None    # ... or fall with it
    final_run_s: float = 20.0
    final_min_intensity: int = 0
    window_s: float = 300.0
    
    
SCARE_AND_RECOVER = SequenceDef(
    name="scare_and_recover",
    phases=(
        Phase("A", goals=("retreat_from:trigger",)),
        Phase("A", goals=("go_to_owner",), until_phases=("C", "R")),
        Phase("C", goals=("bark_at:trigger", "investigate:trigger")),
        Phase("C", goals=("look_to_owner",), until_intensity=3),
        Phase("R", goals=("investigate:toy",)),
        Phase("R", goals=("chew:toy", "play_with:toy", "steal:toy", "zoomies"), within_s=10, after=("A", "C")),
        Phase("P", goals=("chew:toy", "play_with:toy", "steal:toy", "zoomies")),
    ),
    optional=("C",),
    intensity={ "investigate:toy": 1, "chew:toy": 2, "play_with:toy": 3, "steal:toy": 4, "zoomies": 4 },
    builds_up_with='play',
    final_min_intensity=3,
)

OWNER_LEAVES = SequenceDef(
    name="owner_leaves",
    phases=(
        Phase("W", ("wait_by_door",)),
        Phase("S", ("chew:toy",)),
        Phase("S", ("explore",)),
        Phase("Z", ("rest",)),
        Phase("Z", ("lie_down",)),
    ),
    intensity={"wait_by_door": 3, "chew:toy": 2, "explore": 2, "rest": 1, "lie_down": 1},
    builds_down_with="panic",
    final_run_s=20, final_min_intensity=1, window_s=600,
)
    
def _dt(times) -> float:
    dt = times[1] - times[0] if len(times) > 1 else 2.0
    if dt <= 0.0:
        raise ValueError(f"[_dt]: sample times must increase, got {times[0]} then {times[1]}")
    return dt

def _intensities(seq, roles) -> dict:
    return { bind(action, roles): value for action, value in seq.intensity.items() }

def _times_from_timeline(timeline) -> list[float]:
    return [t[0] for t in timeline]

    
def bind(goal, roles) -> str:
    parts = list(goal.partition(":"))
    if parts[2] in roles:
        parts[2] = roles[parts[2]]
    return "".join(parts)
        
def phase_order(seq)-> list[str]:
    res = []
    phases = list(seq.phases)
    for p in phases:
        if p.name not in res:
            res.append(p.name)
    return res

def classify(seq, timeline, roles) -> list[str | None]:
    # starting values
    res = []
    phases_seen = set()
    highest_intensity = 0
    last_seen = {}
    rows = []
    intensities =  _intensities(seq, roles)

    # 1. we need a bind loop appends to a list that keeps Phases and bound goals together.
    for p in seq.phases:
        bounded_goals = [bind(goal, roles) for goal in list(p.goals)] 
        rows.append((p, bounded_goals))
        
    # 2. Loop over timeline
    for t, goal in timeline:
        label = None
        # 3. Loop over the phase and bounded goals contained in rows
        for phase, bounded_goals in rows:
            # Rule 1: If the current goal is not in bounded_goals, fail
            if goal not in bounded_goals:
                continue
            # Rule 2: If any phase in until_phases has already been seen, fail
            if any(name in phase.until_phases for name in phases_seen):
                continue
            # Rule 3: If the phase's intensity limit is less than or equal to the highest intensity seen, fail
            if phase.until_intensity is not None and phase.until_intensity <= highest_intensity:
                continue
            # Rule 4
            if phase.within_s is not None:
                # collect the times for the phases listed in phase.after filtering for those only in last_seen
                times = [last_seen[p] for p in phase.after if p in last_seen]
                # 4A: if no prior times, fail.
                if len(times) == 0:
                    continue
                # 4B: if the latest recorded time - current time is greater than the within_s range, fail
                if t - max(times) > phase.within_s:
                    continue
            # Result: This passes all 4 rules, record the phase name
            label = phase.name
            break
        # 4. Append the label to the result array.
        res.append(label)
        
        # 5. update tracking values:
        highest_intensity = max(intensities.get(goal, 0), highest_intensity)
        if label is not None:
            last_seen[label] = t
            phases_seen.add(label)
    return res

def first_reached(phases, times) -> dict[str, float]:
    res = {}
    if len(phases) != len(times):
        raise ValueError("[first_reached]: phases and times array lengths differ")
    comp = zip(phases, times)
    for p, t in comp:
        if p is not None and p not in res:
            res[p] = t
    return res

# run-length encoding
def bouts(phases, times) -> list[tuple[str | None, float, float]]:
    if not phases or not times:
        return []
    if len(phases) != len(times):
        raise ValueError("[bouts]: phases and times array length differ")
    res = []
    start_time = times[0]
    cur_phase = phases[0]
    sample_count = 1
    dt = _dt(times)
    
    for i in range(1, len(phases)):
        if phases[i] == cur_phase:
            sample_count += 1
        else:
            end_time = start_time + sample_count * dt
            res.append((cur_phase, start_time, end_time))
            start_time = times[i]
            cur_phase = phases[i]
            sample_count = 1
    res.append((cur_phase, start_time, start_time + sample_count * dt))
    return res

def in_order(seq, first) -> bool:
    phase_ord = phase_order(seq)
    missing = False
    last_time = float("-inf")
    
    for phase in phase_ord:
        if phase not in first and phase not in seq.optional:
            missing = True
        if phase in first:
            time = first[phase]
            if time <= last_time or missing:
                return False
            last_time = time
    return True

def time_in(phases, times) -> dict[str, float]:
    b = bouts(phases, times)
    res = {}
    for p, s, e in b:
        if p is not None:
            res[p] = res.get(p, 0) + (e - s)
    return res

def first_bout_s(seq, phases, times) -> float:
    first_phase = phase_order(seq)[0]
    b = bouts(phases, times)
    for p, s, e in b:
        if p == first_phase:
            return e - s
    return 0.0

def backsteps(seq, phases, times, triggers=()) -> list[tuple[float, int, bool]]:
    res = []
    order = {o:i for i, o in enumerate(phase_order(seq))}
    prev, furthest = -1, -1
    for p,t in zip(phases,times):
        if p is None:
            continue
        # backstep
        if order[p] < prev:
            res.append((t, furthest - order[p], any(t-4 <= trigger <= t for trigger in triggers)))
        prev = order[p]
        furthest = max(furthest, order[p])
    return res

def final_run_start(seq, timeline, phases, roles) -> float | None:
    final_run_s = seq.final_run_s
    times = _times_from_timeline(timeline)
    dt = _dt(times)
    n = round(final_run_s / dt)
    final_phase = phase_order(seq)[-1]
    
    streak = 0
    run_start = None
    intensities = _intensities(seq, roles)
    
    for ((t, goal), label) in zip(timeline, phases):
        if label == final_phase and intensities.get(goal, 0) >= seq.final_min_intensity:
            if streak == 0:
                run_start = t
            streak += 1
            if streak == n:
                return run_start
        else:
            streak = 0
    return None

def window_shares(phases, times, phase, start, end, window_s = 20.0) -> list[float]:
    low = start
    windows = []
    while low < end:
        high = min(low+window_s, end)
        window = [p for p,t in zip(phases, times) if low <= t < high]
        if len(window) != 0:
            share = window.count(phase) / len(window)
            windows.append(share)
        low+=window_s
    return windows


def spearman(x, y) -> float:
    def rank_list(data) -> list[float]:
        n = len(data)
        indexed = sorted(enumerate(data), key=lambda item: item[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2.0 # endpoint midpoint average formula
            # Assign the average rank back to the original positions
            for k in range(i, j + 1):
                original_idx = indexed[k][0]
                ranks[original_idx] = avg_rank
            i = j + 1
        return ranks
    
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if all(i == x[0] for i in x) or all(i == y[0] for i in y):
        return 0.0
    rank_x, rank_y = rank_list(x), rank_list(y)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])
    
def follows_emotion(seq, timeline, phases, roles, emotion, triggers=()) -> tuple[float | None, int]:
    if len(phase_order(seq)) < 2:
        return (None, 0)
    times =  _times_from_timeline(timeline)
    dt = _dt(times)
    reached = first_reached(phases, times)
    second_to_last = phase_order(seq)[-2]
    if seq.builds_up_with is None and seq.builds_down_with is None or emotion is None or second_to_last not in reached:
        return (None, 0)

    second_to_last_time = reached[second_to_last]
    frs = final_run_start(seq, timeline, phases, roles)
    final_run_end = None
    if frs is None:
        final_run_end = times[-1] + dt
    else:
        final_run_end = frs + seq.final_run_s
    
    intensities =  _intensities(seq, roles)
    if len(emotion) != len(timeline):
        raise ValueError("emotion and timeline array lengths differ")
    samples = [(t, intensities[goal], e) for (t, goal), e in zip(timeline, emotion) if goal in intensities and final_run_end > t >= second_to_last_time]
    rho = None
    if len(samples) >= 5:
        rho = spearman([s[1] for s in samples], [s[2] for s in samples])
    steps_against = 0
    for s_1, s_2 in zip(samples, samples[1:]):
        jump = s_2[1] - s_1[1]
        if seq.builds_up_with:
            bad = jump < -1                      # dropped by more than 1
        else:
            bad = jump > 1                       # rose by more than 1
        if bad and not any(s_2[0]-4 <= trigger <= s_2[0] for trigger in triggers):
            steps_against += 1
    return (rho, steps_against)

def measure_run(seq, timeline, roles, emotion=None, triggers=()) -> dict:
    if len(timeline) == 0:
        raise ValueError("[measure_run]: timeline is empty")
    phases = classify(seq, timeline, roles)
    times = _times_from_timeline(timeline)
    dt = _dt(times)
    first_reach = first_reached(phases, times)
    order = phase_order(seq)
    reaction_latency = first_reach.get(order[0])
    fbs = first_bout_s(seq, phases, times)
    ti = time_in(phases, times)
    io = in_order(seq, first_reach)
    backstps = backsteps(seq, phases, times, triggers=triggers)
    final_start = final_run_start(seq, timeline, phases, roles)
    check_shares = []
    if len(order) >= 3 and order[-2] in first_reach :
        check_shares = window_shares(phases, times, order[-3], first_reach[order[-2]], final_start if final_start is not None else times[-1] + dt)
 
    rho, steps_against = follows_emotion(seq, timeline, phases, roles, emotion, triggers=triggers)
    switches = 0
    prev_goal = timeline[0][1]
    for _, g in timeline:
        if g != prev_goal:
            switches += 1
        prev_goal = g

    return {
        'phases': phases,
        'first_reached': first_reach,
        'reaction_latency': reaction_latency,
        'first_bout_s': fbs,
        'time_in': ti,
        'in_order': io,
        'backsteps': backstps,
        'check_shares': check_shares,
        'final_start': final_start,
        'rho': rho,
        'steps_against': steps_against,
        'switches': switches
    }