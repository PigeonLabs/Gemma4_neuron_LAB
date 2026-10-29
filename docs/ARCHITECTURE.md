# Gemma 4 12B Unified 아키텍처 감사

감사일: 2026-10-29 (Asia/Seoul). **이 문서는 원격 config·공식 소스의 정적 감사다. 가중치 로딩, BF16 출력 동등성, GPU 실행은 여기서 검증한 것으로 취급하지 않는다.** 실제 로딩 시 module 이름·shape·dtype를 재확인해야 한다.

## 확인한 모델과 버전

| 항목 | 확인값 |
|---|---|
| 모델 ID | `google/gemma-4-12B-it` |
| HF revision | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| 공개 / 접근 제한 | HF API: `private=false`, `gated=false`, `disabled=false` |
| architecture / model_type | `Gemma4UnifiedForConditionalGeneration` / `gemma4_unified` |
| 라이선스 | Apache-2.0 (모델 메타데이터·공식 카드) |
| config 저장 당시 Transformers 버전 | `5.10.0.dev0` — 실행 최소 버전을 뜻하지 않음 |
| 조사한 upstream commit | `huggingface/transformers@144ecf007b8b4f2910da104c84a79801db6985a6` |
| 조사한 안정 release | `v5.21` (release 본문: 5.21.0, 2026-10-26) |
| modeling 파일 blob | `fdd10ec0737454f01fd0c6f4ad3072becc1739b8`; 위 main commit과 release가 동일 |

근거: [HF 모델 API](https://huggingface.co/api/models/google/gemma-4-12B-it), [공식 config](https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/config.json), [안정 release](https://github.com/huggingface/transformers/releases/tag/v5.21).

`gemma4` 일반 계열, Gemma 3, Qwen 계열과 `gemma4_unified`는 별개 구현이다. 자동 클래스 지원을 이름만으로 추정하지 않는다. 실제 설치본에서 해당 config 클래스와 generation 클래스를 import하고 모델 로딩을 확인해야 한다.

## Text backbone

| 항목 | config 기준 |
|---|---:|
| decoder 계층 | 48 |
| residual hidden dimension | 3,840 |
| dense MLP intermediate dimension | 15,360 |
| MLP 중간 좌표 수 (48 × 15,360) | 737,280 |
| activation | `gelu_pytorch_tanh` |
| vocabulary | 262,144 |
| 최대 position | 262,144 |
| sliding window | 1,024 |
| query heads | 16 |
| local head dimension / KV heads | 256 / 8 |
| global head dimension / KV heads | 512 / 1 |
| RMS epsilon | 1e-6 |
| final logit softcap | 30 |

위 뉴런 수는 **config로부터 산출한 예상치**다. 실제 로딩한 모든 `down_proj.in_features` 합으로 확정한다. checkpoint는 `use_double_wide_mlp=false`, `num_kv_shared_layers=0`, `enable_moe_block=false`로 설정되어 있다. 일부 config 클래스의 기본값(30 계층 등)을 checkpoint 값과 혼동하면 안 된다.

계층은 sliding 5개 + full 1개의 패턴을 8회 반복한다. 0부터 세면 full attention은 L5, L11, L17, L23, L29, L35, L41, L47이다. global attention은 `attention_k_eq_v=true` 조건에 따라 K projection 출력을 V의 원재료로도 사용한다. K와 V의 normalization은 구별되므로 최종 K·V tensor가 같다는 뜻은 아니다. local RoPE theta는 10,000, global proportional RoPE theta는 1,000,000이며 global rotary fraction은 0.25다. Text-only 입력은 causal attention을 사용한다.

근거: [checkpoint config](https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/config.json), [attention 구현 L331](https://github.com/huggingface/transformers/blob/144ecf007b8b4f2910da104c84a79801db6985a6/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py#L331), [config 변환 구현](https://github.com/huggingface/transformers/blob/144ecf007b8b4f2910da104c84a79801db6985a6/src/transformers/models/gemma4_unified/configuration_gemma4_unified.py).

## 뉴런의 정의와 정확한 hook 위치

분석할 뉴런 `(layer, neuron)`은 **gate activation과 up projection의 원소별 곱을 계산한 뒤, down projection에 들어가기 직전 중간 tensor의 마지막 축 좌표**다. 단순 gate 출력이나 residual coordinate, SAE feature와는 구별한다.

수식 (linear weight 표기):

```text
x = RMSNorm_pre_ffn(residual_after_attention)
a = GELU_tanh(W_gate x) ⊙ (W_up x)
m = W_down a
r_out = (r_in + RMSNorm_post_ffn(m)) × layer_scalar
```

`a`의 일반적 shape는 `[batch, sequence, 15360]`이다. `a[..., n]`이 뉴런 n의 활성값이다. 세 projection은 bias가 없다. `nn.Linear` 모듈의 `forward_pre_hook`를 **down projection**에 등록하면 이 tensor를 직접 관찰·변경할 수 있다.

공식 conditional generation wrapper의 경로는 다음과 같다.

```text
model.model.language_model.layers[L].mlp.down_proj
```

여기서 맨 앞 `model`은 사용자가 생성한 Python 변수다. `named_modules()` 문자열은 `model.language_model.layers.L.mlp.down_proj`가 예상된다. `Gemma4UnifiedForCausalLM` 단독 text wrapper는 `model.model.layers[L].mlp.down_proj` 구조이므로 자동 어댑터에서 wrapper를 확인해야 한다.

**중요한 인과 해석 제약:** down projection 뒤에 `post_feedforward_layernorm`이 있다. 한 좌표를 더해도 residual에 더해지는 최종 벡터 변화는 일반적으로 `delta × W_down[:, n]`과 같지 않다. 다른 좌표의 크기와 post normalization이 개입 효과에 영향을 준다. 단일 뉴런 steering과 residual vector steering을 동등한 조작으로 보고하면 안 된다.

근거: [MLP L447–463 및 decoder L466–515](https://github.com/huggingface/transformers/blob/144ecf007b8b4f2910da104c84a79801db6985a6/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py#L447), [multimodal wrapper L1275](https://github.com/huggingface/transformers/blob/144ecf007b8b4f2910da104c84a79801db6985a6/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py#L1275).

### Hook 실행 계약

- 관측 hook은 입력을 수정하지 않고 `None`을 반환한다. 저장 시 detach하고, 장기 보존은 요약 통계·제한된 top-k로 한정한다.
- 개입 hook은 입력 tensor를 clone하고 지정한 좌표·token mask만 수정한 tuple을 반환한다. dtype·device·shape를 유지한다.
- additive `a_n ← a_n + delta`, multiplicative `a_n ← a_n × scale`, zero `a_n ← 0`, replacement `a_n ← value`, restoration `a_n ← matched_baseline_n`를 분리한다.
- restoration은 baseline과 입력·token position·layer가 정렬된 값을 사용한다. baseline과 intervention은 각각 새 KV cache로 시작한다.
- capture 범위(prefill/decode/양쪽), token mask(전체/마지막/내용 token), 개입 seed와 강도를 Run에 기록한다. padding은 집계에서 제외한다.
- handle은 `finally` 또는 context manager에서 제거한다. `torch.compile`, tracing backend, NNsight 호환성을 정적 구조만으로 PASS로 표시하지 않는다.

## Chat, tokenizer, prefill/decode

공식 예제의 기본 인터페이스는 `AutoProcessor` + `AutoModelForMultimodalLM`다. `processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)`를 사용해 thinking 조건을 고정한다. 모델과 processor는 동일 HF revision으로 로딩하고 template 문자열의 hash를 기록한다. 수동으로 BOS를 중복 삽입하지 않는다.

공식 템플릿은 `system`을 지원하고 assistant를 내부 `model` role로 바꾼다. 모델의 원본 `generation_config.json`에는 sampling이 활성화되어 있다. 따라서 smoke parity·hook 제거·동일 조건 재현 검증에는 `do_sample=False`와 `eval()`·`inference_mode()`를 명시한다. seed만 고정한 채 원래 sampling 설정을 두는 방법은 smoke의 결정적 baseline을 보장하지 않는다.

prefill은 전체 prompt token을 처리하고, cache가 활성화된 이후 decode는 새 token들을 처리한다. 두 단계 모두 같은 MLP/down projection을 통과한다. 모델은 cache 길이를 반영해 position IDs를 계산한다. generation의 `prepare_inputs_for_generation`은 첫 단계 이후 multimodal token type mask를 제거한다. 길이가 1인 tensor만으로 prefill/decode를 판별하지 말고 실행 단계 또는 첫 forward 여부를 별도로 추적한다. 반복 generation은 이전 run의 cache를 재사용하지 않는다.

근거: [공식 모델 사용법](https://huggingface.co/google/gemma-4-12B-it#usage), [고정 revision generation config](https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/generation_config.json), [TextModel L565](https://github.com/huggingface/transformers/blob/144ecf007b8b4f2910da104c84a79801db6985a6/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py#L565), [generation L1411](https://github.com/huggingface/transformers/blob/144ecf007b8b4f2910da104c84a79801db6985a6/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py#L1411).

## 멀티모달 범위

12B Unified는 별도 vision/audio encoder tower 대신 raw image patches·audio waveform frame을 LM embedding 공간에 투영한다. 이미지 전처리는 16×16 patch를 3×3 병합하고, audio는 16kHz의 640 sample frame을 사용한다. 현재 연구는 text-only이므로 projection의 내부 뉴런과 입력 모달리티 비교는 측정 대상에서 제외한다. 다른 Gemma 4 크기의 encoder 구조를 이 모델에 대입하면 안 된다.

근거: [공식 Transformers Unified 문서](https://github.com/huggingface/transformers/blob/v5.21/docs/source/en/model_doc/gemma4_unified.md), [Gemma4UnifiedModel L868](https://github.com/huggingface/transformers/blob/144ecf007b8b4f2910da104c84a79801db6985a6/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py#L868).

## 검증 상태

| 검증 | 상태 | 근거/잔여 조건 |
|---|---|---|
| 모델 ID·공개 접근·revision 확인 | PASS | 공식 HF API 메타데이터 |
| config의 뉴런 정의·예상 dimension 확인 | PASS | checkpoint config와 공식 MLP source 대조 |
| 안정 release의 클래스·hook 위치 확인 | PASS | v5.21 source; 실제 설치/실행 검증과 구별 |
| 실제 모델 BF16 로딩 | NOT_EVALUATED | runtime smoke 결과로 판정 |
| 실제 48개 hook tensor shape 확인 | NOT_EVALUATED | runtime capture 필요 |
| read-only parity·개입 수식·제거 복원 | NOT_EVALUATED | runtime smoke 및 unit test 필요 |
| 모델 기능·뉴런 인과 효과 | NOT_EVALUATED | discovery/validation/test 실험 필요 |

이 표는 정적 감사 시점의 상태다. 실행 결과는 해당 Run의 요약 및 연구 로그를 우선한다. 새 runtime 측정이 생기면 commit·패키지 버전과 함께 이 문서를 갱신한다.
