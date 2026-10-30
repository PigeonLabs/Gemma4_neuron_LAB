"""Second, separately registered readout repair; never modifies measured R10."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from neuron_lab import round10_repair as original
from neuron_lab.round10_repair import parse_final, parse_generated_body, generation_boundary, option_audit

ROOT = Path(__file__).resolve().parents[2]
STRICT_FORMAT = (
    'Answer this question using its option code. '
    'Output one plain ASCII line in the form FINAL:<code>, '
    'replacing <code> with A or B. Output no other text.'
)


def build_manifest(cfg, old):
    combined = deepcopy(old)
    prior = json.loads((ROOT/'runs/round10_manifest.json').read_text())
    for study in cfg['studies']:
        for split, worlds in prior['studies'][study]['splits'].items():
            combined['studies'][study]['splits']['R10_'+split] = worlds
    manifest = original.build_manifest(cfg, combined)
    manifest['round'] = 'R10b_direct_final_repair'
    for detail in manifest['studies'].values():
        for worlds in detail['splits'].values():
            for w in worlds:
                w['id'] = w['id'].replace('R10-', 'R10b-', 1)
    return manifest


def prepared_cases(manifest, study, split, role):
    result = []
    for case in original.prepared_cases(manifest, study, split, role):
        stem, old_format = case['text'].rsplit('\n\n', 1)
        assert old_format == original.FORMAT
        if study == 'A' and role == 'anchor':
            stem = stem.replace('Does the player know the exact state before choosing the action?',
                'Under the stated event order, is the exact state DIRECTLY ANNOUNCED to the player '
                'before the action is chosen? Consider the direct announcement only, not information inferred from a signal.')
        result.append({**case, 'text': stem+'\n\n'+STRICT_FORMAT})
    if split == 'calibration':
        # Fixed deterministic round robin spreads worlds and elementary kinds
        # across an early futility stop. No generated answer controls ordering.
        grouped = defaultdict(list)
        for case in result:
            grouped[(case['factors'].get('anchor_kind', 'sentinel'), case['episode_id'])].append(case)
        keys = sorted(grouped)
        result = [grouped[k][i] for i in range(max(map(len, grouped.values())))
                  for k in keys if i < len(grouped[k])]
    yield from result


def calibration_futility(observed, planned, cfg):
    """Only declared elementary readout bounds, never main semantic accuracy."""
    expected = Counter(c['factors']['anchor_kind'] for c in planned)
    details = {}
    failed = []
    for kind, n in expected.items():
        rows = [r for r in observed if r['factors']['anchor_kind'] == kind]
        missing = sum(r['semantic_choice'] is None for r in rows)
        not_confirmed = sum(r['semantic_choice'] != r['truth'] for r in rows)
        upper_parse = (n-missing)/n
        upper_correct = (n-not_confirmed)/n
        threshold = cfg['code_anchor_accuracy_min'] if kind == 'code' else cfg['anchor_accuracy_min']
        details[kind] = {'planned': n, 'observed': len(rows),
            'best_possible_parse_fraction': upper_parse,
            'best_possible_confirmed_correct_fraction': upper_correct}
        if upper_parse < cfg['anchor_parse_min'] or upper_correct < threshold:
            failed.append(kind)
    return {'stop': bool(failed), 'failed_groups': failed, 'groups': details,
            'scope': 'Engineering calibration futility only; no scientific Negative and no main-effect selection.'}
