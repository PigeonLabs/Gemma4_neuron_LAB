# 고정 연구의 재현과 새로운 Discovery의 구분

이 저장소의 첫 연구 주기는 discovery에서 선택한 네 좌표와 가설·반례를 고정한 뒤 독립 validation/test를 측정했습니다. 실행 조건은 `configs/model.yaml`, `configs/hypotheses.json`, `configs/validation_cases.json`에 있습니다. 고정한 모델 revision과 discovery summary SHA256을 바꾸면 원래 연구와 같은 confirmatory replay가 아닙니다.

## 공개 discovery 근거

[`catalog/public/frozen_evidence.json`](../catalog/public/frozen_evidence.json)은 원래 로컬 `runs/<survey_run_id>/candidates.json`에서 **이미 선택·등록된 네 좌표의 측정 record를 그대로 복사한 subset**입니다. 다시 순위를 매기거나 통계를 계산하거나 validation/test로 후보를 고르지 않았습니다. 공개 top-48 목록은 그대로 유지하며, 그 목록에 없는 고정 후보의 범주별 프로필도 별도 subset에서 재현할 수 있습니다.

이 파일은 모델 revision, 원래 survey run ID, 등록 discovery summary SHA256, 비공개 전체 후보 파일 SHA256을 기록합니다. 분석기는 네 좌표가 정확히 일치하고 revision·run ID·등록된 `std`, `min`, `max`, `selectivity_z`가 같은지 검사합니다. 원래 full artifact가 있으면 그것을 우선 사용하고, 없으면 이 공개 subset을 사용합니다. 분석 출력인 `analysis.json`이나 `neurons.json`을 측정 입력으로 읽지 않습니다.

원래 전체 후보 파일이 남아 있는 연구 환경에서는 다음 명령으로 같은 subset을 내보낼 수 있습니다. 기존 subset과 원본이 충돌하면 덮어쓰지 않고 실패합니다.

```bash
.venv/bin/python scripts/analyze_results.py --export-frozen-evidence
```

## Confirmatory causal replay

프로젝트 루트에서 고정 의존성을 설치하고 모델을 다운로드합니다. 모델 가중치와 캐시는 공개 Git 저장소에 포함하지 않습니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python scripts/prepare_model.py
```

원래 연구의 `logs/summary/survey.json`과 고정 가설·반례를 보존한 상태에서 다음 작업을 순서대로 제출합니다. 단일 GPU worker는 먼저 smoke 검증을 통과해야 다음 실험을 수행합니다.

```bash
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit smoke
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit steering
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit validation
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli submit expanded
PYTHONPATH=src .venv/bin/python -m neuron_lab.cli worker
```

이 경로는 고정 좌표에 대한 새 causal run을 만들며, 새 run ID와 실제 측정값을 기록합니다. BF16·드라이버·커널 등의 환경 차이까지 독립적으로 검증한 것은 아니므로 다른 환경에서 bitwise 동일한 모델 출력을 보장하지 않습니다.

완료한 run의 해시와 자료를 검사하여 분석·그림·README를 재생성합니다.

```bash
.venv/bin/python scripts/analyze_results.py
.venv/bin/python scripts/analyze_results.py --check
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_steering_figures.py
.venv/bin/python scripts/build_readme.py
.venv/bin/python scripts/build_readme.py --check
.venv/bin/python -m pytest -q
```

원본 interventions JSONL과 activation observations는 로컬 자료입니다. 공개 summary만으로 그림은 재생성할 수 있지만, 원시 개입을 재검사하는 `analyze_results.py`에는 해시가 일치하는 해당 run 자료 또는 위 명령으로 새로 실행한 자료가 필요합니다. 관련 paraphrase는 독립 표본으로 취급하지 않습니다.

## 공개 파일로부터의 CPU 분석 점검

```bash
.venv/bin/python scripts/check_reproduction.py
```

이 검사는 `/tmp`에 공개 허용 목록의 파일을 복사하고, 완료한 causal run의 `metadata.json`, `interventions.jsonl`, `activation_observations.json`만 추가합니다. 모델 가중치와 원래 survey 전체 후보 파일은 복사하지 않습니다. 파생 출력인 analysis·neuron catalog도 분석 전에 제거합니다.

이 임시 복사본에서 `compute(root=임시경로)`를 실행하여 공개 frozen subset이 실제로 사용됐는지 확인하고, 네 후보의 AUC·causal signature·반례 분석·활성 프로필·프로필 간 관계와 전체 과학적 출력이 원본 환경의 CPU 계산과 같은지 검사합니다. 출처 경로와 출처 해시 receipt의 차이만 비교에서 제외합니다. 검사가 끝나면 임시 디렉터리를 지우고 [`logs/summary/reproduction.json`](../logs/summary/reproduction.json)에 영수증을 기록합니다.

아직 완료하지 않은 stage는 `NOT_EVALUATED`로 남습니다. 이후 test run이 완료되면 이 검사를 다시 실행해야 그 stage도 재현 점검에 포함됩니다. 이 점검은 데이터 의존성과 CPU 분석의 재현성을 입증하며, 추가 GPU 추론이나 독립 모델 반복 실험을 실행하지 않습니다.

## 새로운 Discovery 연구

새 pilot/survey는 별도의 탐색 주기입니다. 원래 고정 연구의 `logs/summary/survey.json`을 덮어쓰거나 기존 가설의 hash를 새 측정에 맞춰 바꾸지 마세요. 새로운 작업 디렉터리나 별도 연구 버전에서 discovery를 실행하고, 새 run·새 summary digest·새 고정 가설·새 독립 검증 자료로 다시 등록해야 합니다. 기존 test 결과를 보고 같은 test에 맞춰 가설을 다시 고르면 confirmatory 검증이 아닙니다.
