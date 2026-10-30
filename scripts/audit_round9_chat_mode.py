#!/usr/bin/env python3
"""CPU-only audit of actual thinking-flag serialization, not a new model test."""
import hashlib
import json
from pathlib import Path
import resource
import time
import yaml
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic();cp=ROOT/'configs/round9_chat_mode_audit.yaml';cfg=yaml.safe_load(cp.read_text())
    reg=json.loads((ROOT/'logs/summary/round9_P0_registration.json').read_text())
    local=Path((ROOT/'runs/model_path.txt').read_text().strip());assert local.name==reg['model_revision']
    tok=AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False)
    model_load=json.loads((ROOT/'logs/summary/round9_P0_model_load.json').read_text())
    template_hash=hashlib.sha256(str(tok.chat_template).encode()).hexdigest()
    assert template_hash==model_load['chat_template_sha256']
    msg=[{'role':'user','content':'Answer only A: is 1 equal to 1?'}]
    false=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    true=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    closed='<|channel>thought\n<channel|>'
    assert false.endswith(closed) and false!=true and '<|think|>' in true
    suffix=tok.encode(closed,add_special_tokens=False);bits={tok.encode(b,add_special_tokens=False)[0] for b in '01'}
    ep=ROOT/'runs/round9_P0_encoded_inputs.json';assert sha(ep)==reg['encoded_inputs_sha256']
    encodings=json.loads(ep.read_text());matched=0;prefix_counts={0:0,1:0,2:0}
    for enc in encodings.values():
        ids=enc['input_ids'];found=[]
        for extra in [0,1,2]:
            if extra and any(i not in bits for i in ids[-extra:]):continue
            end=len(ids)-extra
            if ids[end-len(suffix):end]==suffix:found.append(extra)
        assert len(found)==1
        matched+=1;prefix_counts[found[0]]+=1
    elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert elapsed<=cfg['max_seconds'] and rss<=cfg['max_RSS_bytes']
    out={'schema_version':1,'status':'PASS','scope':'CPU tokenizer/input serialization audit only','new_model_forwards':0,
         'registered_actual_model_forwards_unchanged':1790,'model_revision':reg['model_revision'],
         'actual_template_sha256':template_hash,'matches_model_load_template':True,'enable_thinking_false_and_true_renderings_differ':True,
         'false_mode_suffix':closed,'true_mode_inserts_system_think_marker':True,
         'registered_inputs_checked':len(encodings),'closed_empty_thought_suffix_matches':matched,
         'appended_binary_prefix_lengths':prefix_counts,'encoded_inputs_sha256':sha(ep),
         'interpretation':'enable_thinking=False was not ignored, and the registered input does not leave an open thought channel. This does not establish why complex tasks start with To or whether later answers are correct.',
         'registered_classifications_changed':False,'CPU_seconds':elapsed,'CPU_peak_RSS_bytes':rss,
         'config_sha256':sha(cp),'script_sha256':sha(Path(__file__))}
    raw=json.dumps(out,ensure_ascii=False,indent=2)+'\n';assert len(raw.encode())<=cfg['max_artifact_bytes']
    (ROOT/'logs/summary/round9_chat_mode_audit.json').write_text(raw)
    print(json.dumps({'status':'PASS','new_forwards':0,'inputs_checked':matched,'flag_ignored':False}))


if __name__=='__main__':main()
