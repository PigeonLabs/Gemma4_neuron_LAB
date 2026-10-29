"""Codex-authored first-cycle hypotheses, frozen before held-out execution."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


CASES = {
    "H001": {
        "positive_control": ["Translate the English word 'cat' into Portuguese. Give one word.", "Translate 'house' from English into Italian. Give one word."],
        "negative_control": ["How many letters are in the English word translate? Give only the integer.", "Define the grammatical term noun in one short English sentence."],
        "near_miss": ["Write the word translation exactly once and stop.", "Is French the name of a language? Answer yes or no without translating anything."],
        "paraphrase": ["What is the Portuguese equivalent of the English animal name cat?", "How would an Italian speaker say the English noun house?"],
        "out_of_distribution": ["스페인어로 우유를 어떻게 말하나요? 단어만 답하세요.", "हिंदी शब्द पानी का अंग्रेज़ी अर्थ क्या है? एक शब्द में उत्तर दें।"],
    },
    "H002": {
        "positive_control": ["A marble was hidden in one of two cups while I was away. I received no clues. Which cup holds it?", "I secretly chose an integer from 1 through 20 and gave you no clues. Which integer did I choose?"],
        "negative_control": ["The label says the drawer contains three red pencils. What color are the pencils?", "Compute 6 plus 5 and give the exact integer."],
        "near_miss": ["Count the letters in the English word uncertain. Give only the count.", "Copy the single English word maybe and stop."],
        "paraphrase": ["Nobody told us whether a hidden marble is under the left cup or the right cup. Can its location be determined?", "Without any evidence about my secret number between one and twenty, can you identify it exactly?"],
        "out_of_distribution": ["불투명한 주머니 안의 공 색은 공개되지 않았습니다. 공은 무슨 색인가요?", "A remote sensor's reading was never recorded or transmitted. What exact value did it show yesterday?"],
    },
    "H003": {
        "positive_control": ["Translate the English word 'bird' into French. Give one word.", "Translate the English adjective 'red' into German. Give one word."],
        "negative_control": ["Name a tool used to measure the length of a desk.", "Give the integer result of 8 minus 3."],
        "near_miss": ["French and German are language names. How many language names are in the preceding sentence?", "Write the English word Spanish exactly once without translating it."],
        "paraphrase": ["How would a French speaker say the English animal name bird?", "What is the German equivalent of the English color adjective red?"],
        "out_of_distribution": ["일본어로 고양이는 무엇인가요? 한 단어만 적으세요.", "¿Cuál es la palabra inglesa para azul? Responde con una palabra."],
    },
    "H004": {
        "positive_control": ["Write a JavaScript function named twice that doubles its numeric argument.", "Write a SQL query returning the title column from a table named novels."],
        "negative_control": ["Describe a small green tree in one English sentence.", "What number comes immediately after eleven? Give only the integer."],
        "near_miss": ["A python is an animal. Name the broad animal group it belongs to without writing computer code.", "Write the word Python in uppercase and stop. Do not provide code."],
        "paraphrase": ["Provide JavaScript source for a function that multiplies an input by two.", "Give SQL source that reads only title values from novels."],
        "out_of_distribution": ["Write a Rust function that returns the sum of two i32 values.", "Write one R expression that computes the mean of c(2,4,6)."],
    },
}


def main():
    destination = ROOT / "configs/hypotheses.json"
    if destination.exists():
        raise SystemExit("Hypotheses already frozen; do not overwrite a registered study")
    survey_path=ROOT/"logs/summary/survey.json"
    survey=json.loads(survey_path.read_text())
    if survey["status"]!="PASS":
        raise ValueError("Requires successful measured discovery survey")
    candidates=json.loads((ROOT/"runs"/survey["run_id"]/"candidates.json").read_text())
    selected=[candidates[0],next(c for c in candidates if c["category"]!="language_translation"),
              next(c for c in candidates if c["layer"]>=24 and c["category"]=="language_translation"),
              next(c for c in candidates if c["category"]=="programming")]
    descriptions=[
        ("번역 과제 문맥에 반응하는 좌표일 수 있다.", ["번역 동사·언어 이름의 표면 토큰", "외국어 문자와 문장 길이"]),
        ("주어진 정보로 답을 결정할 수 없는 문맥에 반응하는 좌표일 수 있다.", ["uncertain/know 등 표면 어휘", "질문·부정문 형태"]),
        ("후기 계층의 번역 과제 문맥 좌표일 수 있다.", ["언어 이름 자체의 표상", "번역 요청의 어휘·구문"]),
        ("프로그래밍 과제 전반에 반응하는 좌표일 수 있다.", ["Python이라는 특정 어휘·동물 이름", "코드 문법·표면 언어 이름"]),
    ]
    hypotheses=[]
    for i,(candidate,(description,competitors)) in enumerate(zip(selected,descriptions),1):
        hypotheses.append({"id":f"H{i:03d}","group":"primary" if i<=2 else "expanded",
            **{k:candidate[k] for k in ["layer","neuron","std","min","max","selectivity_z","top_contexts","low_contexts"]},
            "target_category":candidate["category"],"hypothesis":description,"competing_hypotheses":competitors,
            "status":"candidate","discovery_evidence_run":survey["run_id"]})
    frozen={"model_revision":survey["model_revision"],"frozen_kst":datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "generator":"Codex evidence review in this session; no external model API",
        "discovery_summary_sha256":hashlib.sha256(survey_path.read_bytes()).hexdigest(),
        "selection_rule":"After discovery only: highest selectivity; highest non-translation; highest translation at layer>=24; highest programming. No validation/test measurements used.",
        "candidates":hypotheses,
        "registered_checks":{
            "correlation":"Evaluate fixed target-vs-rest prompt-mean AUC on validation and untouched test; exploratory threshold AUC>=0.8 in both. Report n=3 positive per category, p values with Holm adjustment across four candidates.",
            "counterexamples":"Positive/paraphrase/OOD versus negative/near-miss activation. Paraphrases share source_id with validation positive originals, never cross splits.",
            "distribution_selectivity":"At +1 SD only, compare (candidate KL - mean projected-direction-control KL) on target vs other prompts; stratified prompt bootstrap95%CI. Require positive lower bound on validation AND test to support only a category-selective distribution effect.",
            "semantic_function":"Not established by KL, AUC, or technical restoration. Require independent behavioral task correctness and confound-matched study before a semantic causal label.",
            "multiple_selection":"737280 discovery coordinates screened; discovery ranking is descriptive, not statistical significance. Four fixed heldout hypotheses are reported, not a rescreened test leaderboard."}}
    destination.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+"\n")
    case_rows=[]
    for candidate in hypotheses:
        for role,texts in CASES[candidate["id"]].items():
            for index,text in enumerate(texts):
                source_role="positive_control" if role=="paraphrase" else role
                case_rows.append({"id":f"counter-{candidate['id']}-{role}-{index}",
                    "source_id":f"counter/{candidate['id']}/{source_role}/{index}","split":"validation",
                    "category":candidate["target_category"] if role in {"positive_control","paraphrase","out_of_distribution"} else "irrelevant_control",
                    "candidate_id":candidate["id"],"role":role,"text":text,
                    "expected_activation_group":"high" if role in {"positive_control","paraphrase","out_of_distribution"} else "low"})
    cases={"hypotheses_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),"prompts":case_rows,
           "scope":"New validation originals with within-validation paraphrase families; corpus test remains untouched."}
    (ROOT/"configs/validation_cases.json").write_text(json.dumps(cases,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"hypotheses":len(hypotheses),"new_validation_cases":len(case_rows),"hypotheses_sha256":cases["hypotheses_sha256"]}))


if __name__=="__main__":
    main()
