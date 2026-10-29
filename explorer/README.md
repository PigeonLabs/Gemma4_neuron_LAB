# 로컬 Neuron Explorer

Streamlit 기본 위젯을 사용하는 연구 콘솔이다. 실제 측정 자료가 없으면 미평가·자료 없음으로 표시하며, 예시 뉴런이나 가상 그래프를 만들지 않는다.

프로젝트 root에서 가상환경을 활성화한 뒤 실행한다.

```bash
python -m streamlit run explorer/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

브라우저에서 [로컬 콘솔](http://127.0.0.1:8501)을 연다. Streamlit 1.39 이상을 권장하며 NumPy를 사용한다. 모델/PyTorch는 콘솔 프로세스에서 불러오지 않는다. 외부 공개 서버가 아닌 loopback 전용 실행이다.

## 자료와 탐색

- `catalog/public/candidates.json`, `catalog/public/neurons.json`: layer/neuron/category/validation status 필터, 활성 통계, 보존한 token 표본의 히스토그램, 범주 평균, 상·하위 문맥. 공개 catalog가 없으면 동결 후보의 실제 필드만 표시한다.
- `configs/hypotheses.json`: 동결한 후보 ID, 가설과 경쟁 설명. 공개 catalog의 검증 상태가 있으면 이를 반영한다.
- `logs/summary/*.json`: 개입 조건별 baseline 대비 집계·CI·표본 수, 분할별 활성 상관, run 상태·시간·GPU memory 및 원본 요약.
- `runs/jobs/*.json`: 최근 로컬 작업의 pending/running/done/failed 상태. 라이브 결과는 해당 브라우저 세션에서 제출한 작업을 자동으로 확인한다.

같은 좌표라도 model revision이 다르면 다른 항목으로 다룬다. 히스토그램은 저장된 `sampled_activations`만 사용하며 quantile이나 평균에서 분포를 복원하지 않는다. 데이터가 없는 지표를 0이나 성공으로 채우지 않는다. Category selectivity·AUC·분포 변화량은 의미 기능이나 행동 정확도 자체가 아니다. 해석 기준은 [방법론](../docs/METHODOLOGY.md)을 따른다.

파일은 화면 새로고침 때 다시 읽는다. 부분 저장·잘못된 JSON·24 MiB를 넘는 표시용 파일은 표시를 건너뛰고 읽기 오류로 남긴다. 모델 가중치·원본 activation을 읽는 기능은 없다.

## Live steering

`logs/summary/smoke.json`의 `status`가 `PASS`이고 `runs/worker.json`의 `status`가 `READY`일 때만 제출 폼이 활성화된다. 동결 후보 목록이 없을 때도 비활성화된다. Worker 실행·모델 로딩·smoke 통과는 CLI의 별도 작업이며 콘솔이 명령을 실행하지 않는다.

동결 후보, 사용자 prompt, `sigma ∈ [-2,2]`, `max_new_tokens ≤ 24`를 선택하면 다음 형식의 로컬 작업을 원자적으로 저장한다.

```json
{"stage":"live","prompt":"사용자가 입력한 문장","candidate_id":"동결 목록의 실제 ID","sigma":1.0,"max_new_tokens":16}
```

이는 설명용 작업 schema이며 측정 결과가 아니다. 실제 파일 이름은 `runs/jobs/<time_ns>_live.pending.json`이다. 단일 GPU worker가 입력 길이·후보·강도를 다시 검증하고 순차 실행하며, `*.done.json`의 `result.baseline`과 `result.intervention` text를 나란히 표시한다. 각 필드는 문자열 또는 `{"text": "…"}` 객체를 허용한다. `*.failed.json`의 오류도 표시한다. 작업 완료 전에는 완료된 출력을 만들어 보여주지 않는다.

Live 요청은 현재 forward의 마지막 token에 additive steering을 적용하는 탐색이다. 입력 최대 길이는 chat template 포함 256 token이며 초과 입력은 worker가 거부한다. Live 비교는 discovery 이후의 임의 탐색이고 독립 validation/test 결과에 자동 포함되지 않는다. 프롬프트와 생성 결과는 Git에서 제외한 로컬 `runs/`에 남는다. API key 입력, 외부 API 호출, 쉘 명령 실행 기능은 없다.
