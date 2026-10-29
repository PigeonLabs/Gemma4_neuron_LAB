# 참조 구현 감사

확인일: 2026-10-29 (Asia/Seoul). GitHub connector로 각 저장소의 실제 `main` 브랜치, recursive tree, 아래 핵심 파일과 LICENSE를 읽었다. Neuronpedia 공식 문서에서 현재 interp-engine 저장소를 찾아 연결했다. **이번 감사는 정적 소스 검토이며 참조 프로젝트는 실행하지 않았다.** upstream의 실험 결과는 본 연구의 실험 결과로 계산하지 않는다. 기계 판독 가능한 조회 목록은 [audit_sources.json](../references/audit_sources.json)에 있다.

## 고정한 참조 버전

| 프로젝트 | 확인한 main SHA | 라이선스 | 본 연구에서 사용할 범위 |
|---|---|---|---|
| [Lyra neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer/tree/6a6f0e95bf49ad6626940021148f775d7d55d1e4) | `6a6f0e95bf49ad6626940021148f775d7d55d1e4` | MIT | Steering-first 설계, 자연 활성 범위와 대조군, 로그·카탈로그 형식 참고 |
| [Apple ml-refusal-neurons](https://github.com/apple-aiml-research/ml-refusal-neurons/tree/b9afae4f25274051ee2a08f9ccb04a70b9dc04cf) | `b9afae4f25274051ee2a08f9ccb04a70b9dc04cf` | Apple 자체 소프트웨어 라이선스 | 단일 좌표 훅 수명, 개입 위치와 평가 분리 참고 |
| [EleutherAI Delphi](https://github.com/EleutherAI/delphi/tree/4fea06e6e8b68eeaf302474325fca13df95c5d6f) | `4fea06e6e8b68eeaf302474325fca13df95c5d6f` | Apache-2.0 | 증거·설명·독립 점수화 분리와 비활성 사례 참고 |
| [interp-engine](https://github.com/decoderesearch/interp-engine/tree/2afe000da14f1fff8b9c611665f7fe343f523dd0) | `2afe000da14f1fff8b9c611665f7fe343f523dd0` | Apache-2.0 | 공통 hook point 정의, backend parity 평가 및 추후 가속 후보 |
| [NNsight](https://github.com/ndif-team/nnsight/tree/de7fec4d207fc8e0deacaa44fb68f8d0412506ef) | `de7fec4d207fc8e0deacaa44fb68f8d0412506ef` | MIT | PyTorch 모듈 입출력 접근과 activation patching 후보 |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens/tree/6bea1fa7cfcbf9064aa8ac99bbd5fb1e4a6c2240) | `6bea1fa7cfcbf9064aa8ac99bbd5fb1e4a6c2240` | MIT | Gemma 4 adapter 및 HF parity 검증 참고 |

LICENSE의 실제 내용을 확인했다. Apple 저장소는 MIT나 Apache로 재표기하면 안 된다. 이번 초기 구현은 이들의 알고리즘과 검증 방법을 참고하는 독립 구현이며 소스를 통째로 복사하거나 외부 가중치·데이터·캐시를 재배포하지 않는다. 나중에 코드 일부를 재사용하면 해당 revision의 라이선스·저작권·NOTICE 요구사항을 함께 보존한다.

## A. Lyra: Steering-first 탐색

실제로 확인한 경로는 `sweep/neuron_stats.py`, `sweep/rank.py`, `sweep/steer_sweep.py`, `sweep/check_equivalence.py`, `explorer/runtime.py`, `explorer/agent_methodology.md`, `NOTES.md`, `research_log.md`이다. `research_log.md`는 tree에서 존재를 확인했으며 이 감사에서 전체 내용의 타당성을 재검증하지 않았다.

- `neuron_stats.py`는 Qwen3.5의 gate/up 값을 보고 `down_proj` 직전의 중간 활성을 수집한다. 자연 활성 절댓값 최대와 down-projection 방향, residual 크기를 개입 단위에 연결한다.
- `steer_sweep.py`는 양·음 dose, baseline, 동일 norm의 무작위 방향을 함께 실행하고 JSONL로 생성 결과를 보존한다. `rank.py`는 단어 분포와 대조군 대비 구별성을 평가한다. 자동 설명을 먼저 확정하는 대신 개입 결과에서 후보를 찾는 설계를 채택할 수 있다.
- `check_equivalence.py`는 HF 좌표 개입과 별도 vLLM fork의 출력 공간 개입을 비교한다. `runtime.py`는 단일 뉴런을 조절하는 `down_proj` pre-hook을 사용한다.
- Explorer는 runtime, server, MCP agent, static UI를 나누며 검색·세 방향 생성 비교·활성 읽기·유사 연결을 제공하도록 작성되어 있다. 기능이 이번 환경에서 정상 실행됨은 확인하지 않았다.

Gemma 4에 Qwen3.5의 SiLU 공식, 모듈 경로, attention 코드, vLLM fork를 그대로 이식하지 않는다. 이 연구에서는 실제 Gemma config와 forward를 확인하고 **`down_proj`에 실제로 입력되는 텐서**를 직접 계측한다. 선형 down-projection 직후에는 좌표 delta와 해당 column을 곱한 delta가 대응하지만, Gemma의 post-MLP normalization을 지난 residual에 동일 벡터를 더하는 방법은 일반적으로 같은 개입이 아니다. Lyra 소스도 post-FFN norm이 있는 모델에 단순 치환을 적용하는 일을 경계한다. 양·음 극성은 서로 다른 문맥 기능일 수 있으므로 활성 절댓값 순위만으로 기능을 단정하지 않는다.

## B. Apple: 단일 뉴런 훅과 분류 평가

실제 tree에서 `hooks/`, `find_refusal_neurons/`, `find_concept_neurons/`, `max_activations/`, `detection/`, `eval/`을 확인했다. 자세히 읽은 구현은 `hooks/model_hooks.py`, `hooks/amplify_hooks.py`, `hooks/neuron_suppressor.py`, `find_concept_neurons/discovery/individual_cv.py`다.

- `model_hooks.py`는 LLaMA/Qwen 형태와 GPT-2 형태의 모듈을 구분하고 pre/down 출력 등 서로 다른 capture 단계를 지원한다. 모델 이름만으로 Gemma 호환을 보증하는 구현이 아니다.
- `amplify_hooks.py`에는 `down_proj` pre-hook 경로와 MLP 전체 재계산 경로가 공존한다. pre-hook 경로는 원래 forward가 이미 계산한 중간 활성에 작용해 projection 중복 계산을 피한다. prefill/decode 적용 위치를 별도로 지정하는 설계가 참고된다.
- `neuron_suppressor.py`는 hook 활성·비활성화와 제거를 분리한다. context manager 종료 시 비활성화되지만 handle 자체는 별도 remove 메서드로 제거하므로, 본 연구에서는 예외를 포함한 handle 제거를 직접 검증한다.
- `individual_cv.py`의 개별 feature CV는 활성값으로 범주를 예측하는 상관 평가다. 높은 분류 정확도를 인과적 역할의 증거로 승격하지 않는다. 원본/패러프레이즈의 group 단위 분할이 별도로 필요하다.

기존 모델용 neuron 좌표와 강도를 Gemma에 적용하지 않는다. 안전성 우회용 preset이나 위해 생성물을 카탈로그에 복제하지 않으며, 초기 검증은 일반 지식·수학·언어·형식 등 무해한 과제로 수행한다. `max_activations`의 DB·query·viewer 분리와 `eval`의 생성/평가 분리는 storage와 evaluation 설계 참고 사항이다. 이 하위 도구들을 이번 환경에서 실행한 것은 아니다.

## C. Delphi: 설명과 반증의 분리

`delphi/latents/`, `delphi/explainers/`, `delphi/scorers/`, `delphi/pipeline.py`의 실제 구조를 확인했고 `delphi/config.py`, `delphi/explainers/explainer.py`, `delphi/scorers/classifier/detection.py`를 읽었다.

- LatentRecord와 활성 사례를 설명 생성기에 전달하고 설명을 점수화 단계에 다시 넣는 구조다. activation quantile·무작위·상위 사례의 sampling 설정과 비활성 예시 소스를 명시한다.
- 설명 문자열·사례·score를 따로 저장하는 구조를 참고해, 본 연구는 경쟁 가설, 양성·음성·near-miss·OOD 사례, 생성 데이터와 검증 데이터의 출처를 함께 기록한다.
- Delphi가 주로 다루는 SAE latent는 학습된 sparse dictionary 좌표다. 여기서 정의한 native MLP 중간 뉴런과 동일하지 않다. SAE 설명 점수가 단일 뉴런 steering의 인과 효과를 대신하지 않는다.
- 설명 모델 서비스 호출에는 별도 API 설정과 비용이 필요할 수 있다. 초기 로컬 파이프라인 검증에 서비스 호출을 필수로 두지 않는다. Codex가 증거를 읽어 만든 가설도 독립 검증 전에는 `candidate`에 머문다.

## D. Neuronpedia interp-engine: 정확한 좌표와 backend 검증

[Neuronpedia 공식 문서](https://docs.neuronpedia.org/open-source/interp-engine)가 현재 연결하는 저장소는 `decoderesearch/interp-engine`이다. 실제로 `interp_engine/address.py`, `interp_engine/arch.py`, `interp_engine/steer.py`, `interp_engine/steer_specs.py`, `validator/`, `tests/`가 존재한다. README, LICENSE, `docs/SUPPORTED_POINTS.md`, `docs/COMPATIBILITY.md`, `docs/ARCHITECTURE_QUIRKS.md`, Gemma 12B cross-engine 결과와 steering/architecture 구현을 읽었다.

`mlp_pre`(gate branch), `mlp_pre_linear`(up branch), `mlp_act`(down-projection 입력), `mlp_out`, `mlp_out_post`를 구분한다. 본 연구의 neuron 정의는 **`mlp_act`**에 대응한다. 단순한 point 이름만 저장하지 말고 실제 모듈 경로·shape·dtype·position 범위를 함께 남겨야 한다. Eager와 vLLM의 제공 point와 fused 모듈 구조는 서로 다르다.

upstream의 [Gemma 12B cross-engine 결과](https://github.com/decoderesearch/interp-engine/blob/2afe000da14f1fff8b9c611665f7fe343f523dd0/validator/comparison/results/google/gemma-4-12B-it/0_result_details.md)는 2×H200에서 layer 0/24/36/47을 평가했다. 이 보고서의 일부 vLLM 중간 `mlp_act` 값은 eager 대비 불일치로 표시되며 NNsight와 새로운 TransformerLens 경로는 해당 비교를 통과했다고 기록한다. **이는 upstream 측정이며 본 연구 GPU의 통과 결과가 아니다.** backend가 지원 목록에 있다는 사실만으로 BF16 parity가 확보되지 않는다.

고정 revision의 README는 Gemma 4/vLLM config 호환 때문에 Transformers 5.14.1을 요구한다고 명시한다. 5.15에서 per-layer attention config가 바뀌는 문제가 이유다. 이 제약을 순수 HF 실행 전반의 필수 버전으로 확대 해석하지 않는다. 처음에는 HF eager + 직접 PyTorch pre-hook으로 기준선을 만들고, 향후 가속 backend를 추가할 때 동일 revision·입력·dtype에 대한 logits/activation/steering parity를 재측정한다.

## E. NNsight / TransformerLens

[NNsight 공식 문서](https://nnsight.net/)와 실제 `src/nnsight/modeling/language.py`를 확인했다. 현재 문서는 `TransformersModel`과 trace를 통해 내부 텐서를 읽고 수정하는 예를 제시한다. LanguageModel은 호환 경로로 남아 있지만 구버전 튜토리얼의 import만 따라가면 최신 API와 어긋날 수 있다. 구조를 직접 확인할 수 있고 activation patching의 후보지만 **이 환경의 Gemma 12B BF16 통과는 NOT_EVALUATED**다.

TransformerLens main의 `transformer_lens/model_bridge/supported_architectures/gemma4.py`는 `Gemma4UnifiedForConditionalGeneration` 12B를 명시적으로 다룬다. 텍스트 decoder는 `model.language_model` 아래에 있고 Unified는 별도 vision encoder 없이 projector를 가진다. `gemma4_text.py`는 LM head를 가진 순수 텍스트 variant의 다른 경로를 구분한다. Gemma 4 RMSNorm에는 Gemma 1–3에서 사용한 weight offset을 그대로 적용하면 안 된다. adapter는 여러 optional 구조를 허용하며 forward 수학을 HF에 위임하는 방식을 채택한다.

`tests/integration/model_bridge/test_gemma4_bridge.py`에는 tiny random Gemma 4 fixture의 logit parity·hook·cache 테스트가 존재한다. tiny fixture 시험의 존재를 실제 12B pretrained BF16 통과로 간주하지 않는다. 현재 README는 `TransformerBridge`를 권장하고 구 `HookedTransformer.from_pretrained` API 제거를 알린다. 가중치의 normalization folding이나 centering을 켜면 native 좌표가 바뀔 수 있어 초기 계측은 원본 HF 수학을 유지한다.

## 초기 구현 결정과 수용 기준

1. 정식 HF 구현과 고정 model revision을 기준선으로 사용한다. 원격 사용자 코드는 자동 신뢰하지 않는다.
2. 실제 `down_proj` 입력을 neuron 좌표로 삼아 capture와 개입을 구현한다. MLP 종류·정확한 shape가 확인되지 않으면 실패로 처리한다.
3. capture-only 출력 불변성, 개입 공식, handle 제거, 재현성을 먼저 검사한다. 모든 호출에서 token mask와 prefill/decode 정책을 기록한다.
4. category selectivity는 후보 선정용이며, dose-response·random neuron·norm-matched random direction·ablation/restoration과 독립 prompt 검증을 거쳐 상태를 올린다.
5. raw activation 전체 대신 streaming 요약과 제한된 evidence만 저장한다. 원본 prompt family 단위로 discovery/validation/test를 분리한다.
6. 외부 engine 가속, UI 대규모 이식, 유료 설명 모델 연동은 기준선이 검증된 다음 진행한다. upstream 측정값은 자체 연구 그래프·뉴런 카탈로그에 넣지 않는다.

이 문서의 참조 호환성은 소스·문서 확인에 근거한 판단이다. 실제 설치·로드·실험의 PASS/FAIL은 각 run의 로그에 별도로 기록한다.
