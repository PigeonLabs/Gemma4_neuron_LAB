# Neuron Explorer 검증 기록

확인일: 2026-10-29 (Asia/Seoul). 대상은 [로컬 연구 콘솔](http://127.0.0.1:8501/)이다. 이 문서는 UI의 실제 표시와 동작을 검증한 기록이며, 뉴런 기능 가설의 검증 결과가 아니다.

## 환경과 대상 흐름

검증 흐름: **로컬 앱 열기 → 측정 catalog 표시 → Layer/Neuron/Category/Validation status 필터 → 뉴런 상세·관련 뉴런·개입 효과 확인 → worker 상태에 따른 Live 폼 확인.**

- Streamlit 1.66.1, Playwright 1.62.1, Chrome 156.0.8078.73.
- Desktop 1440×1000, mobile viewport 390×844.
- Browser plugin not available. 세션에 Browser plugin/skill이 없어 bundled regular Playwright와 이미 설치된 `/opt/google/chrome/chrome`을 사용했다. 기본 Playwright browser executable은 없어서 설치된 Chrome 경로를 지정했다.
- Sandbox 안 Chrome 시작은 `setsockopt: Operation not permitted`로 실패했다. 승인된 sandbox 밖 실행으로 같은 loopback 주소를 검사했다. 브라우저나 GPU 의존성을 새로 설치하지 않았다.
- 실측 `catalog/public/`, `configs/hypotheses.json`, `logs/summary/`를 읽었다. UI 회귀시험에서 임시 상태 파일은 pytest 전용 임시 디렉터리에 격리했다.

## 확인 결과

| 검사 | 결과 | 증거 |
|---|---|---|
| Page identity | PASS | URL `http://127.0.0.1:8501/`, title `Gemma 4 Neuron Lab · Research Console` 일치 |
| 첫 화면이 비어 있지 않음 | PASS | 실제 catalog table·뉴런 선택기·활성 히스토그램 표시 |
| Framework/runtime 오류 화면 | PASS | Streamlit exception overlay 0개, AppTest exception 0개 |
| 브라우저 console | PASS with warning | JavaScript/page error 0개. 아래 Vega scale 경고 존재 |
| 네 종류의 필터 | PASS | L15 → N9637 → language_translation → ambiguous에서 H001 1건과 같은 상세 좌표 표시 |
| 관련 뉴런 | PASS | H001의 실제 관련 항목 3개를 ID·좌표·Pearson r·범주 수·run ID 표로 확인 |
| 개입 효과 | PASS | `js_divergence` 선택 후 동일 metric caption과 측정 집계 그래프·표 표시 |
| 가설·검증/실행 기록 | PASS | 동결 가설·분할별 활성 상관·run 요약 탭이 렌더링됨 |
| Live BUSY 상태 | PASS | Smoke PASS + worker BUSY에서 프롬프트·제출 버튼 비활성 |
| Live READY 상태 | PASS | 이후 worker READY 전환 시 제한된 입력 폼 활성. 이 QA에서는 실제 GPU 작업을 제출하지 않음 |
| Mobile viewport | PASS | 문서·body 너비 모두 390px로 전체 페이지 가로 넘침 없음. 표는 컴포넌트 내부 스크롤 사용 |
| Python/UI 회귀시험 | PASS | `tests/test_explorer.py`: 9 passed |

Vega 경고는 `Scale bindings are currently only supported for scales with unbinned, continuous domains.`다. Streamlit 기본 `bar_chart`가 범주 축에도 scale binding을 붙여 발생하며, 데이터 조회·필터·막대 표시를 막지 않았다. 범주 축 zoom은 지원된다고 주장하지 않는다. 경고를 숨기거나 그래프를 제거하지 않았다.

## 최초 timeout의 원인과 변경

초기 Playwright script는 Layer 다중선택에서 `15`를 누른 직후 뉴런 상세 combobox를 찾다가 timeout됐다. Streamlit의 다중선택 dropdown은 항목을 골라도 열린 채 유지되고, React Aria는 열려 있는 popup 밖 요소를 접근성 트리에서 숨긴다. **필터 데이터 손상이 아니라 자동화 script의 dropdown 닫기 누락**이었다. Escape로 popup을 닫고 rerender 후 표시 좌표를 기다리면 L15의 두 옵션만 보였다. 긴 Neuron 목록은 가상화되므로 combobox 검색으로 `9637`을 입력한 뒤 옵션을 선택했다.

`explorer/app.py`의 관련 뉴런 표시를 작은 표로 바꾸고 관찰적 profile 상관이 연결·상호작용·공유 기능을 입증하지 않는다는 설명을 추가했다. 개입 그래프에는 현재 metric 이름을 명시해 조건 선택 후 수치를 읽기 쉽게 했다. 지표를 바꾼 직후에는 이전 그래프가 rerender 중 잠시 남을 수 있어 새 caption과 렌더링 완료를 확인한 뒤 스크린샷을 촬영했다.

회귀시험은 다음을 확인한다.

- 서로 다른 model revision의 같은 좌표를 병합하지 않으며, 공개 catalog의 최신 검증 상태가 동결 당시 상태를 갱신함.
- missing/corrupt/과대 JSON을 실제 측정값이나 0으로 대체하지 않음.
- 실제 catalog의 네 종류 필터가 표와 상세 좌표에 동일하게 적용되고 관련 뉴런 ID가 저장 자료와 일치함.
- 표시용 Vega chart의 Arrow 데이터가 선택한 실제 run/candidate/metric 집계와 일치함.
- Smoke PASS와 worker READY가 동시에 만족돼야 Live 폼을 활성화함.
- 임시 전용 root의 제출 폼이 정해진 범위의 pending JSON을 원자 저장함. 실제 `runs/jobs/`에 쓰거나 GPU를 실행하는 시험이 아님.

## 재실행과 증거

회귀시험 명령은 프로젝트 root에서 `.venv/bin/python -m pytest tests/test_explorer.py -q`다. 브라우저 검증은 기존 서버에 대해 bundled Node/Playwright로 `chromium.launch({executablePath:'/opt/google/chrome/chrome', headless:true})`, loopback `goto`, 역할 기반 필터·탭 조작, DOM 상태 확인, console 수집, desktop/mobile screenshot 순서로 실행했다.

검증 세션의 스크린샷과 임시 script는 `/tmp`에 보존하며 Git 업로드 대상에 넣지 않는다. 최종 화면 증거는 `neuron_explorer_desktop.png`, `neuron_explorer_filtered.png`, `neuron_explorer_related.png`, `neuron_explorer_effects.png`, `neuron_explorer_live_ready.png`, `neuron_explorer_mobile.png`다. 최종 브라우저 결과는 `neuron_explorer_qa_fixed.json`, script는 `neuron_explorer_qa_fixed.cjs`다. 앞선 BUSY 상태 검사는 동일 세션의 console/상태 기록과 AppTest로 확인했다. 이 임시 증거는 연구 결과 데이터의 대체물이 아니다.

## 남은 범위

실제 Live 작업의 대기열 소비·GPU 생성·완료 결과 전달은 주 실험 담당자가 worker가 idle일 때 별도로 확인한다. 이 UI QA만으로 해당 GPU 왕복을 PASS로 표시하지 않는다. Firefox/WebKit, 실제 휴대전화 터치·키보드, 외부 공개 배포, 긴 시간의 worker liveness와 stale READY 파일은 검사하지 않았다. READY 파일은 상태 신호이며 PID heartbeat의 완전한 생존 확인은 아니다. Browser plugin을 설치하면 같은 검증을 앱 안 navigation·DOM·console·screenshot 흐름으로 수행할 수 있다.

## 실제 Live 왕복 추가 검증

2026-10-29 KST, READY worker에서 H001 +1SD, 최대12token, 무해한 산술 입력을 실제 UI로 1회 제출했다. pending→done 작업과 Baseline/Intervention 표시를 확인했다. 두 결과는 모두13이며 JS/pageerror는0이었다. 브라우저 증거 `/tmp/neuron_explorer_live_roundtrip.png`, 실행 receipt `/tmp/neuron_live_roundtrip.json`. 이 임의 Live 입력은 validation/test에서 제외했다.
