"""CPU oracles and planning checks for R5; this module runs no model.

The generated episodes are design examples, not registered confirmation data.
Fresh confirmed manifests must be sealed after discovery-only selection.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def composed_answer(first: dict, second: dict, query: str) -> str:
    return second[first[query]]


def direct_answer(direct: dict, query: str) -> str:
    return direct[query]


def make_episode(seed: int, episode_id: str) -> dict:
    rng = random.Random(seed)
    xs = [f"x{x:02}" for x in rng.sample(range(100), 16)]
    ys = [f"y{x:02}" for x in rng.sample(range(100), 16)]
    zs = [f"z{x}" for x in rng.sample(range(10), 4)]
    rng.shuffle(ys)
    first = dict(zip(xs, ys))
    # Each answer has two train queries and two completely disjoint test queries.
    groups = [xs[4*i:4*i+4] for i in range(4)]
    second = {first[x]: zs[i] for i, group in enumerate(groups) for x in group}
    direct = {x: zs[i] for i, group in enumerate(groups) for x in group}
    cases = {}
    for phase in ("train", "test"):
        selected = [x for group in groups for x in (group[:2] if phase == "train" else group[2:])]
        rng.shuffle(selected)
        correct_codes = list("ABCD") * 2
        rng.shuffle(correct_codes)
        phase_cases = []
        for i, (query, code) in enumerate(zip(selected, correct_codes)):
            other_z = [z for z in zs if z != direct[query]]
            other_codes = [c for c in "ABCD" if c != code]
            rng.shuffle(other_z)
            codebook = dict(zip(other_codes, other_z))
            codebook[code] = direct[query]
            codebook = {c: codebook[c] for c in "ABCD"}
            phase_cases.append({"id": f"{episode_id}-{phase}-{i}", "query": query,
                                "codebook": codebook, "correct_code": code,
                                "answer": direct[query]})
        cases[phase] = phase_cases
    orders = {"first": list(first), "second": list(second), "direct": list(direct)}
    for order in orders.values():
        rng.shuffle(order)
    episode = {"id": episode_id, "seed": seed, "first": first, "second": second,
               "direct": direct, "orders": orders, **cases}
    validate_episode(episode)
    return episode


def validate_episode(episode: dict) -> None:
    first, second, direct = (episode[k] for k in ("first", "second", "direct"))
    if len(first) != 16 or len(second) != 16 or len(direct) != 16:
        raise ValueError("Expected 16-key tables")
    for query in first:
        if composed_answer(first, second, query) != direct_answer(direct, query):
            raise ValueError("The two operational routes disagree")
    if {c["query"] for c in episode["train"]} & {c["query"] for c in episode["test"]}:
        raise ValueError("Training/test query overlap")
    for phase in ("train", "test"):
        rows = episode[phase]
        if len(rows) != 8 or Counter(c["correct_code"] for c in rows) != Counter(dict.fromkeys("ABCD", 2)):
            raise ValueError("Unbalanced response codes")
        if sorted(Counter(c["answer"] for c in rows).values()) != [2] * 4:
            raise ValueError("Unbalanced semantic answers")
        for c in rows:
            if set(c["codebook"]) != set("ABCD") or len(set(c["codebook"].values())) != 4:
                raise ValueError("Invalid codebook")
            if c["codebook"][c["correct_code"]] != composed_answer(first, second, c["query"]):
                raise ValueError("Incorrect oracle code")


def table_text(episode: dict) -> str:
    def pairs(key):
        table = episode[key]
        return ",".join(f"{a}>{table[a]}" for a in episode["orders"][key])
    return ("Use only the supplied tables. f maps X to Y, g maps Y to Z, and h maps X to Z. "
            "h agrees with g(f(X)). A reference is supplied by the evaluator, not an answer you previously produced.\n"
            f"f: {pairs('first')}\ng: {pairs('second')}\nh: {pairs('direct')}")


def query_text(case: dict) -> str:
    codes = " ".join(f"{a}:{b}" for a, b in case["codebook"].items())
    return f"X={case['query']}; {codes}. Reply with one code A/B/C/D."


def reference_record(case: dict, status: str) -> str:
    if status not in {"0", "1", "2"}:
        raise ValueError("Status must be 0/1/2 (incorrect/correct/unreported)")
    return f"{query_text(case)} Reference={case['correct_code']}; diagnostic-pass={status}."


def history_text(episode: dict, statuses: list[str]) -> str:
    if len(statuses) != len(episode["train"]):
        raise ValueError("History length does not match training set")
    return "\n".join(reference_record(c, s) for c, s in zip(episode["train"], statuses))


def permute_statuses(statuses: list[str], strata: list[str], seed: int) -> list[str]:
    """Preserve count in each predeclared stratum, including an identity shuffle."""
    if len(statuses) != len(strata) or any(s not in {"0", "1"} for s in statuses):
        raise ValueError("Invalid status permutation input")
    rng, result = random.Random(seed), list(statuses)
    for group in sorted(set(strata)):
        positions = [i for i, s in enumerate(strata) if s == group]
        values = [statuses[i] for i in positions]
        rng.shuffle(values)
        for i, value in zip(positions, values):
            result[i] = value
    return result


def paired_behavior(scores: dict[tuple[str, str], float]) -> dict[str, float]:
    """One episode, all query scores already averaged with equal weights."""
    recovery = .5 * sum(scores[l, l] - scores[l, "demo"] for l in "AB")
    specificity = .5 * (scores["A", "A"] - scores["A", "B"]
                         + scores["B", "B"] - scores["B", "A"])
    truthful = .5 * sum(scores[l, l] - scores[l, l + "perm"] for l in "AB")
    return {"recovery_vs_demo": recovery, "lesion_history_crossover": specificity,
            "truthful_vs_permuted": truthful}


def paired_recruitment(damage: dict[tuple[str, str, str], float], controls: dict[str, list[str]]) -> float:
    """Own-minus-demo second-ablation interaction minus fixed random-panel mean."""
    terms = []
    for lesion in "AB":
        own = damage[lesion, lesion, "R" + lesion] - damage[lesion, "demo", "R" + lesion]
        panel = controls[lesion]
        if not panel:
            raise ValueError("A random-control panel is required")
        control = sum(damage[lesion, lesion, g] - damage[lesion, "demo", g] for g in panel) / len(panel)
        terms.append(own - control)
    return sum(terms) / 2


def forward_budget(stages: list[dict]) -> dict:
    """No silent factor for response tokens, histories, norm replay or controls."""
    total, rows = 0, []
    for stage in stages:
        components = stage["components"]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in components.values()):
            raise ValueError("Forward budgets must be nonnegative integer calls")
        subtotal = sum(components.values())
        total += subtotal
        rows.append({"stage": stage["id"], "components": components, "planned_actual_forward_calls": subtotal})
    return {"stages": rows, "planned_actual_forward_calls": total,
            "measurement_status": "NOT_EVALUATED", "arithmetic_only": True}
