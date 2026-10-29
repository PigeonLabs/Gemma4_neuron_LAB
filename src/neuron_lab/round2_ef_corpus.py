"""New programmatic instances and held-out prompt shells for R2-E/F.

The inferential unit is a generated instance conditional on these templates,
not an independently authored language task or a representative human prompt.
"""
from __future__ import annotations

from collections import Counter
import json
import random
import string

from .corpus import build_corpus, normalize_text

COLORS = ("blue", "green", "red", "gold", "pink", "white", "black", "brown")


def nonce(rng, used):
    while True:
        value = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 7)))
        if value not in used:
            used.add(value)
            return value


def choices(core, option1, option0, correct_one, order):
    options = [option1, option0] if order == 0 else [option0, option1]
    correct = ("A" if correct_one else "B") if order == 0 else ("B" if correct_one else "A")
    return (core + f"\nA: {options[0]}\nB: {options[1]}\nReply with only A or B.", correct,
            "A" if order == 0 else "B")


def build_manifest(cfg):
    rng = random.Random(cfg["corpus_seed"])
    used, rows = set(), []
    for split in ("calibration", "confirmation"):
        for study in ("E", "F"):
            for index in range(cfg[study][f"{split}_families"]):
                family = f"r2{study}-{split}-{index:03d}"
                if study == "F" or index % 4 == 0:
                    words = [nonce(rng, used) for _ in range(4)]
                    colors = rng.sample(COLORS, 4)
                    mapping = "; ".join(f"{w} means {c}" for w, c in zip(words, colors))
                    q = index % 4
                    op1, op0 = colors[q], words[q]
                    modes = ("translate", "copy") if study == "F" else ("translate",)
                    task = "provided_translation"
                    def core(mode):
                        action = "translate the query into its English meaning" if mode == "translate" else "copy the query exactly"
                        if split == "calibration":
                            return f"Mini-dictionary for the invented language Luma: {mapping}.\nQuery: {words[q]}. Task: {action}. Select the matching result."
                        return f"Use this supplied glossary for the invented language Tavi: {mapping}.\nFor the entry {words[q]}, {action}. Choose the resulting text."
                elif index % 4 == 1:
                    s = nonce(rng, used)
                    while s[::-1] == "".join(sorted(s)):
                        s = nonce(rng, used)
                    operation = index % 3
                    task, modes = "string_rule", ("solve",)
                    op1 = s[::-1] if operation == 0 else "".join(sorted(s)) if operation == 1 else s[1:] + s[:1]
                    op0 = "".join(sorted(s)) if operation == 0 else s[::-1]
                    if op0 == op1:
                        op0 = s.upper()
                    rule = ("reverse the character order", "sort the characters alphabetically", "move the first character to the end")[operation]
                    def core(mode):
                        return (f"String operation: {rule}. Input: {s}. Find the exact output." if split == "calibration" else
                                f"Apply this text rule to {s}: {rule}. Which proposed string is correct?")
                elif index % 4 == 2:
                    name, p, q = [nonce(rng, used) for _ in range(3)]
                    entailed = (index // 4) % 2 == 0
                    given = p if entailed else q
                    task, modes, op1, op0 = "rule_entailment", ("solve",), "entailed", "not entailed"
                    def core(mode):
                        prefix = "Rules and fact" if split == "calibration" else "Consider this closed logical system"
                        return f"{prefix}: every {p} is a {q}; {name} is a {given}. Is '{name} is a {p}' entailed? Use only these statements."
                    # If given p, the query is directly entailed; q alone does not imply p.
                else:
                    names = [nonce(rng, used) for _ in range(3)]
                    colors = rng.sample(COLORS, 2)
                    known = (index // 4) % 2 == 0
                    query = names[0] if known else names[2]
                    task, modes, op1, op0 = "provided_facts", ("solve",), colors[0], "UNKNOWN"
                    def core(mode):
                        prefix = "Supplied facts only" if split == "calibration" else "Treat the following as the complete database"
                        return f"{prefix}: {names[0]} has color {colors[0]}; {names[1]} has color {colors[1]}. What is the color of {query}? Choose UNKNOWN if it was not supplied."
                for mode in modes:
                    correct_one = mode == "translate" if task == "provided_translation" else entailed if task == "rule_entailment" else known if task == "provided_facts" else True
                    for order in range(2):
                        text, correct, semantic1 = choices(core(mode), op1, op0, correct_one, order)
                        rows.append({"id": f"{family}-{mode}-o{order}", "family_id": family,
                                     "split": split, "study": study, "task": task, "mode": mode,
                                     "order": order, "text": text, "correct_label": correct,
                                     "semantic1_label": semantic1, "option1": op1, "option0": op0})
    validate_manifest(rows, cfg)
    return {"schema_version": 1, "seed": cfg["corpus_seed"], "cases": rows,
            "inference_unit": "generated source instance conditional on authored templates",
            "split_templates_differ": True, "external_factual_knowledge_required": False}


def validate_manifest(rows, cfg):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate case IDs")
    if len({normalize_text(r["text"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate normalized prompt texts")
    original = {normalize_text(r["text"]) for r in build_corpus()}
    from .runtime import ROOT
    for path, field in [(ROOT / "configs/validation_cases.json", "prompts"), (ROOT / "configs/utility.yaml", "cases")]:
        if path.is_file():
            if path.suffix == ".yaml":
                import yaml
                other = yaml.safe_load(path.read_text())
            else:
                other = json.loads(path.read_text())
            original.update(normalize_text(r["text"]) for r in other[field])
    partitions = {}
    for r in rows:
        if normalize_text(r["text"]) in original:
            raise ValueError("Round 1 prompt overlap")
        if r["correct_label"] not in {"A", "B"} or r["semantic1_label"] not in {"A", "B"}:
            raise ValueError("Bad oracle label")
        if r["option1"] == r["option0"]:
            raise ValueError("Identical options")
        prior = partitions.setdefault(r["family_id"], r["split"])
        if prior != r["split"]:
            raise ValueError("Family leakage across splits")
    for split in ("calibration", "confirmation"):
        for study in ("E", "F"):
            subset = [r for r in rows if r["study"] == study and r["split"] == split]
            n = cfg[study][f"{split}_families"]
            if len({r["family_id"] for r in subset}) != n or len(subset) != n * (4 if study == "F" else 2):
                raise ValueError("Unexpected family/variant count")
            if Counter(r["correct_label"] for r in subset)["A"] != len(subset) // 2:
                raise ValueError("Unbalanced answer labels")
