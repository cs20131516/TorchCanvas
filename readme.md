# TorchCanvas — 설계 및 로드맵 v0.1

> 한줄 소개: "Drag-and-drop graph designer for PyTorch models"
> 패키지명 제안: `torchcanvas` · CLI: `tcanvas`


## 0. TL;DR
- **목표**: GUI로 블록을 드래그앤드롭 → JSON → PyTorch `nn.Module` 자동 컴파일.
- **핵심 구성**: `LayerFactory`(블록 등록) · `NodeSpec Registry`(입출력/스키마 검증) · `GraphDef`(JSON) · `GraphCompiler`(DAG 실행) · Export(TorchScript/ONNX) · 템플릿 · 간단 GUI(Gradio/Streamlit).
- **우선순위**: (1) 컴파일러 안정화 → (2) 커스텀 블록 팔레트화 → (3) Export/시각화 → (4) GUI → (5) 템플릿.

---

## 1. 목표 & 비범위
### 1.1 목표
- 블록 기반(Conv/Residual/SE/Attention/RNN 등) 신경망을 **코드 작성 없이** GUI로 구성하고, **JSON 스펙**만으로 재현 가능.
- JSON을 **PyTorch 모듈로 컴파일**하고, **학습·배포 친화적** 포맷(TorchScript/ONNX)으로 내보내기.

### 1.2 비범위(초기)
- 분산 학습 오케스트레이션, AutoML, NAS는 제외.
- 완전한 브라우저 골드-스탠다드 UX(캔버스 줌/미니맵 등)는 MVP 이후.

---

## 2. 시스템 구성
```
[GUI(Gradio/Streamlit)]  -- JSON I/O -->  [GraphDef(JSON)]
                                         ↘
                               [NodeSpec Registry] -- 검증 규칙
                                         ↘
                                     [GraphCompiler]
                                 ↙            ↘
                        [PyTorch nn.Module]   [Exporters]
                                            (TorchScript/ONNX)
```

- **LayerFactory**: 블록 생성기. 커스텀 블록(`SEBlock` 등)도 등록 가능.
- **NodeSpec Registry**: 각 블록의 입력/출력 **레이아웃(BCT/BTH)**, 랭크, 파라미터 스키마를 선언. GUI·컴파일러가 공통으로 참조.
- **GraphDef(JSON)**: 노드/간선/입출력 정의. 버전 필수.
- **GraphCompiler**: DAG 토폴로지 정렬, shape 전파, 필요 시 **자동 Permute 삽입**, RNN 인터페이스 표준화.
- **Exporters**: `to_torchscript()`, `to_onnx()` 구현. Netron으로 구조 확인 가이드.

---

## 3. JSON 스키마 v0.1 (초안)
```jsonc
{
  "version": "0.1",
  "metadata": {"name":"HybridCMI-Proto","tags":["cmi","1d"]},
  "nodes": [
    {"id":"imu_in","type":"Input","params":{"shape":[32,256],"layout":"BCT"}},
    {"id":"res1","type":"ResidualSECNNBlock","params":{"in_channels":32,"out_channels":64,"kernel_size":3,"dropout":0.3}},
    {"id":"res2","type":"ResidualSECNNBlock","params":{"in_channels":64,"out_channels":128,"kernel_size":5,"dropout":0.3}},
    {"id":"toBTH","type":"Permute_BCT_to_BTH","params":{}},
    {"id":"gru","type":"GRUBlock","params":{"hidden_size":128,"bidirectional":true,"out":"last"}},
    {"id":"attn","type":"AttentionLayer","params":{"hidden_dim":256}},
    {"id":"fc1","type":"Linear","params":{"in_features":256,"out_features":128,"bias":false}},
    {"id":"bn1","type":"BatchNorm1d","params":{"num_features":128}},
    {"id":"drop1","type":"Dropout","params":{"p":0.3}},
    {"id":"clf","type":"Linear","params":{"in_features":128,"out_features":18}}
  ],
  "edges": [["imu_in","res1"],["res1","res2"],["res2","toBTH"],["toBTH","gru"],["gru","attn"],["attn","fc1"],["fc1","bn1"],["bn1","drop1"],["drop1","clf"]],
  "inputs": ["imu_in"],
  "outputs": ["clf"]
}
```

- `version`은 향후 호환성 보장용.
- `params`는 NodeSpec의 스키마로 검증.

---

## 4. NodeSpec Registry 예시
```jsonc
{
  "ResidualSECNNBlock": {
    "input":  {"rank":3, "layout":"BCT", "shapeHints":"(B,C,T)"},
    "output": {"rank":3, "layout":"BCT"},
    "params": {
      "in_channels":"int>0",
      "out_channels":"int>0",
      "kernel_size":"int>=1",
      "pool_size":"int>=1?=2",
      "dropout":"0.0<=float<=1.0?=0.3",
      "weight_decay":"float>=0?=1e-4"
    }
  },
  "GRUBlock": {
    "input":  {"rank":3, "layout":"BTH", "shapeHints":"(B,T,H)"},
    "output": {"rank":2, "layout":"BH"},
    "params": {
      "hidden_size":"int>0",
      "num_layers":"int>=1?=1",
      "bidirectional":"bool?=true",
      "out":"enum[last|mean|max]?=last"
    }
  },
  "Permute_BCT_to_BTH": {
    "input":  {"rank":3, "layout":"BCT"},
    "output": {"rank":3, "layout":"BTH"},
    "params": {}
  },
  "AttentionLayer": {
    "input":  {"rank":3, "layout":"BTH"},
    "output": {"rank":2, "layout":"BH"},
    "params": {"hidden_dim":"int>0"}
  }
}
```
- `?=`는 기본값 의미.
- GUI는 이 스펙을 이용해 **연결 가능 여부**와 **파라미터 폼**을 자동 생성.

---

## 5. 검증 규칙
- **DAG**: Kahn 알고리즘으로 사이클 검출.
- **레이아웃/랭크**: 연결 양 끝의 레이아웃이 다르면 자동 **Permute 삽입** 옵션 제공.
- **shape 전파**: 최소 `in_channels/in_features` 추론, 미확정이면 첫 유효 텐서에서 lazy-init.
- **RNN/Attention 인터페이스**: `*Block` 래핑으로 출력 텐서를 **단일**로 정규화.
- **오류 메시지**: 노드 id와 파라미터 키를 포함해 원인/가이드를 제공.

---

## 6. Export (TorchScript/ONNX)
- `to_torchscript(model, example_inputs)`
- `to_onnx(model, example_inputs, opset=17)`
- 주의 연산: `expand_as` → ONNX에서 `expand`로 대체 권장, `AdaptiveAvgPool1d` 호환성 확인.
- Netron으로 시각 확인 가이드 동봉.

---

## 7. 테스트 전략
- **유닛**: 각 노드 랭크/레이아웃/파라미터 검증, Permute·Concat 동작.
- **통합**: 단일·브랜치·스킵·머지·시퀀스 그래프 실행.
- **회귀**: 프리셋 JSON(예: HybridCMI) 컴파일→런→Export 스냅샷 비교.

---

## 8. 성능/엔지니어링
- 가능한 **사전 shape 추론**으로 모듈 구성, 필요 시 **lazy build** 유지.
- 학습 루프 플러그인: AMP/GradScaler, EMA, Checkpoint, EarlyStopping hook.
- 대형 모델에서 텐서 라우팅 캐시/메모리 사용량 추적 옵션.

---

## 9. 마일스톤 & 수용 기준
### M1: 컴파일러 안정화
- 필수 노드(Conv1d/BN/ReLU/Pool1d/Linear/Dropout/Concat/Add/Permute/GRUBlock).
- DAG/스키마 검증 통과, 1D 분류 예제 학습 가능. **모든 유닛 테스트 통과**.

### M2: 커스텀 블록 팔레트화
- `SEBlock/ResNetSEBlock/ResidualSECNNBlock/AttentionLayer` 등록.
- HybridCMI JSON 템플릿 컴파일·단일 배치 학습·Export 성공.

### M3: Export & 시각화
- TorchScript/ONNX Export API 제공, Netron 확인 문서.

### M4: 간단 GUI
- 팔레트/캔버스/속성 패널/검증/컴파일 버튼. JSON 저장/불러오기.

### M5: 템플릿 & 프리셋
- Two-Branch + BiLSTM + Attention, UNet-1D, Transformer-Encoder-1D 프리셋.

---

## 10. 리스크 & 대응
- **동적 치수 추론 실패** → NodeSpec로 기대치 명시 + 자동 Permute 삽입.
- **ONNX 비호환 연산** → 변환기 옵션 고정/대체 연산 도입.
- **GUI 복잡도 증가** → MVP는 최소 기능(추가/삭제/선/속성)로 제한.

---

## 11. 예제: HybridCMI 템플릿(JSON 요약)
- 입력(IMU, TOF+Thermal) → IMU ResidualSE · TOF Conv 스택 → Concat → Permute → GRUBlock → Attention → Dense×2 → Classifier.
- 상세 JSON은 레포 예시 폴더에 `templates/hybrid_cmi.json`으로 포함.

---

## 12. 다음 액션 (체크리스트)
- [ ] `Permute_*`, `GRUBlock` 구현 및 등록
- [ ] `NodeSpec Registry` 초안 작성, 스키마 검증기(pydantic/jsonschema)
- [ ] `SEBlock/ResNetSEBlock/ResidualSECNNBlock/AttentionLayer` 등록
- [ ] HybridCMI 템플릿 JSON 작성 → 컴파일 → 학습 스모크 테스트
- [ ] Export API(ONNX/TS) 스텁 작성 → Netron 확인
- [ ] 간단 GUI 초안(Gradio) → 팔레트/캔버스/속성/검증/컴파일

---

### 부록 A. 코드 스텁 (발췌)
- `Permute_BCT_to_BTH`, `Permute_BTH_to_BCT`
- `GRUBlock(out={last|mean|max})`
- Export 함수 시그니처 예시

> 전체 예시는 레포 `core/`와 `examples/`에 배치 예정.

