"""CPU input audit, including discovered A anchor contradiction; never edits studies."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import yaml
from transformers import AutoTokenizer
from neuron_lab.round6_engine import encode
from neuron_lab.round7_tools import token_positions
from neuron_lab import round7_carrier as B,round7_history as C,round7_order_priming as D,round7_transfer as E
from neuron_lab.round7_rule_footprint import case as A_case,rule

def main():
    a=json.loads((ROOT/'runs/round7_A_P0_manifest.json').read_text());bad=[]
    for w in a['splits']['P0']:
        for r in (0,1):
            for qi in (0,2):
                c=A_case(w,qi,w['seed']%2,anchor=r)
                if rule(w['evidence'],w['rules'][r])!=rule(w['evidence'],w['rules'][0]):bad.append(c['id'])
    assert len(bad)==48
    tok=AutoTokenizer.from_pretrained(Path((ROOT/'runs/model_path.txt').read_text().strip()),local_files_only=True)
    counts={};checks={}
    for letter,mod in [('B',B),('C',C),('D',D),('E',E)]:
        cfg=yaml.safe_load((ROOT/f'configs/round7_{letter}.yaml').read_text());m=json.loads((ROOT/f'runs/round7_{letter}_P0_manifest.json').read_text());assert m['splits']==mod.build_manifest(cfg)['splits']
        counts[letter]=len(list(mod.prepared_cases(m,cfg)))
        if letter=='B':
            align=0
            for w in m['splits']['P0']:
                for timing in ['early','late']:
                    for distance in ['short','long']:
                        for q in [0,1]:
                            cases=[B.material(w,q,0,timing,distance,d) for d in [None,'same','other']];ee=[encode(tok,c['text'],1600) for c in cases]
                            spans=[token_positions(tok,c['text'],'CARRIER_BEGIN:\n','\nCARRIER_END') for c in cases]
                            assert len({e['tokens'] for e in ee})==1 and all(len(x)==1 for x in spans) and spans[0]==spans[1]==spans[2]
                            # All source differences occur before the carrier; suffix token IDs identical.
                            assert ee[0]['input_ids'][spans[0][-1]+1:]==ee[1]['input_ids'][spans[1][-1]+1:]==ee[2]['input_ids'][spans[2][-1]+1:];align+=1
            checks['B_donor_alignment_cases']=align;checks['B_graph_qualification']=B.graph_qualification()
        if letter=='C':
            for w in m['splits']['P0']:
                for q in [0,1]:
                    for bi in [0,1]:
                        cs=[C.material(w,q,bi,t,h,o) for t in ['direct','compose'] for h in ['related1','related2','unrelated1','unrelated2'] for o in ['before','after']]
                        assert len({c['oracle']['final_text'] for c in cs})==1 and len({c['expected'] for c in cs})==1
            checks['C_final_state_equality']='PASS all P0 cases'
        if letter=='D':
            n=0
            for w in m['splits']['P0']:
                for q in [0,1]:
                    for bi in [0,1]:
                        for task in ['forward','inverse']:
                            for placement in ['before','after']:
                                vv=[]
                                for o in ['ascending','descending']:
                                    c=D.material(w,q,bi,task,o,placement);e=encode(tok,c['text'],1600);ix=token_positions(tok,c['text'],'PREFIX_BEGIN:\n','\nPREFIX_END');vv.append(sorted(e['input_ids'][i] for i in ix))
                                assert vv[0]==vv[1];n+=1
            checks['D_exact_token_multiset_pairs']=n
        if letter=='E':
            ids=[w['signature'] for x in m['splits'].values() for w in x];assert len(ids)==len(set(ids));checks['E_disjoint_worlds']=len(ids)
    out={'schema_version':1,'status':'PASS_WITH_RECORDED_A_PROTOCOL_DEFECT','new_model_forwards':0,'A_anchor_inconsistent_rows':48,'A_affected_rows_sha256':hashlib.sha256(json.dumps(bad).encode()).hexdigest(),
         'A_interpretation':'R1 anchor scores cannot assess pure rule competence; original registered stop remains. Unconflicted R0 near17/24 and far8/24 also below original .85 gate. No posthoc rerun or exclusion.',
         'B_C_D_E_prepared_counts':counts,'CPU_checks':checks,'all_previous_outcomes_preserved':True}
    (ROOT/'logs/summary/round7_input_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
