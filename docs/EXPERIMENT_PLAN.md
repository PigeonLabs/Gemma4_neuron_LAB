# 첫 번째 제한된 연구 주기

1. 공식 revision의 BF16 모델을 단일 GPU에 로드한다. config 감사와 실제 모듈 경로를 대조한다.
2. read-only hook/logits/greedy generation 동등성, 개입 수식, 제거 후 복원, 반복 재현을 검증한다.
3. 15개 무해한 범주, discovery/validation/test의 독립 source_id를 가진 자체 작성 입력으로 시작한다.
4. 선택 계층 pilot을 측정한 뒤 전체 48계층 streaming 통계를 수집한다. prompt 길이 영향을 줄이기 위해 prompt-balanced 통계를 별도로 기록한다.
5. discovery만으로 후보를 정하고 경쟁 가설 및 반례를 동결한다.
6. 양/음 개입, ablation, restoration, 무작위 뉴런·방향 대조군을 비교한다. 자연 활성값의 표준편차 단위로 강도를 정한다.
7. validation/test에서 분포 변화, greedy 응답 변화, 반복/붕괴, 범주 선택성을 평가한다. 행동 기능을 입증하지 못하면 ambiguous로 남긴다.
8. 첫 주기에서 얻은 측정과 한계를 CP8 보고서로 정리한다. 장기·멀티모달 확장은 새 예산과 연구 질문으로 남긴다.

합성 단위 테스트는 구현 정확성만 평가한다. 소규모 프롬프트 집합은 모집단 대표 표본이 아니다. 토큰을 독립 관측치로 간주한 신뢰구간을 만들지 않는다. prompt 단위 paired bootstrap과 시드별 재현성을 사용한다.
