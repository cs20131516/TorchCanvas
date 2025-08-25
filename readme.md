TorchCanvas

ComfyUI 감성으로 PyTorch 모델을 설계하고, 버튼 한 번에 학습 가능한 소스 코드(nn.Module)를 뽑는다.
원하면 논문 블록(Residual/SE/VGGBlock 등)으로 그룹(컴포지트) 해 재사용.

목차

배경

핵심 기능

빠른 시작

그래프 JSON 스펙

컴포지트(논문) 블록

검증 규칙 요약

아키텍처

템플릿 예시

로드맵

리포 구조

테스트 전략

라이선스

배경

대부분의 GUI 툴은 ONNX/TorchScript(배포용) 아티팩트에 초점을 둡니다. TorchCanvas는 개발·학습용 PyTorch 코드 Export를 1차 산출물로 삼아, 내 프로젝트에 바로 가져가 학습/커스터마이즈할 수 있게 합니다.

핵심 기능

🎛️ 드래그·드롭으로 블록(Conv/BN/ReLU/Pool/Concat/Add/Permute/GRU…)을 배치·연결

🧠 컴포지트(논문) 블록: SEBlock, ResidualBlock, VGGBlock 등을 단일 블록으로 사용

🧾 코드 미리보기 & 다운로드: 그래프(JSON) → 순수 PyTorch 코드(단일 파일 or ZIP 패키지)

🧪 더미 학습 스크립트로 전·후방향 + 옵티마이저 한 바퀴 스모크 테스트

✅ 검증기(NODE_SPECS): 토폴로지/랭크/머지 조건/파라미터 점검 및 자동 제안

Status

Core codegen(그래프→nn.Module): ✅

Lazy layers(LazyConv/LazyLinear/LazyBN): ✅

1D/시퀀스(Conv1d/GRU/Permute): ✅

Merge: Concat ✅ | Add ⬜

Validation(NODE_SPECS): ⬜

Composite blocks(SE/Residual/VGG): ⬜

GUI: Streamlit 미니 UI ✅ | React 정식 DnD ⬜

Export: PyTorch 코드 ✅ | ZIP 패키지 ⬜ | (선택) ONNX/TorchScript ⬜

Templates(VGG/ResNet/HybridCMI): ⬜

Tests(unit/integration/E2E): ⬜

빠른 시작
A) 미니 GUI (Streamlit)
pip install streamlit torch
streamlit run app.py
# 브라우저에서 노드 추가 → 엣지 연결 → inputs/outputs 지정 → 코드 다운로드

B) 코드 생성기 CLI
python torchcanvas_codegen.py --graph graph.json --name MyPkg --out ./generated --with-train
# 결과: generated/MyPkg/{__init__.py, blocks.py, model.py, (옵션)train.py}

C) 더미 학습 스모크
python train_dummy.py --pkg generated.MyPkg --pkg-path . --C 9 --T 256 --num-classes 18 --epochs 1

그래프 JSON 스펙
{
  "version": "0.2",
  "metadata": {"name": "my_model", "tags": ["demo"]},
  "nodes": [
    {"id":"inp","type":"Input","params":{}},
    {"id":"conv","type":"Conv1d","params":{"out_channels":64,"kernel_size":5,"padding":"same"}},
    {"id":"relu","type":"ReLU","params":{}},
    {"id":"perm","type":"Permute_BCT_to_BTH","params":{}},
    {"id":"gru","type":"GRUBlock","params":{"hidden_size":128,"out":"last"}},
    {"id":"fc","type":"Linear","params":{"out_features":18}}
  ],
  "edges": [["inp","conv"],["conv","relu"],["relu","perm"],["perm","gru"],["gru","fc"]],
  "inputs": ["inp"],
  "outputs": ["fc"]
}


type은 원자(Conv/BN/…) 또는 컴포지트(SEBlock/Residual/VGGBlock)

params는 팔레트 기본값을 덮어씀

다중 입력 노드(Concat/Add)는 in-edges가 2개 이상

컴포지트(논문) 블록
내부 정의(ASCII)
[SEBlock]
  in → GAP(1) → Linear(C→C/r) → ReLU → Linear(C/r→C) → Sigmoid
       → scale × in → out

[ResidualBlock]
  in → Conv(k,outC) → BN → ReLU → Conv(k,outC) → BN ──┐
        └─── proj(1×1) if C≠outC ─────────────────────┤ (+) → ReLU → out
short ─────────────────────────────────────────────────┘

[VGGBlock]
  in → Conv(k) → ReLU → [LRN?] → Conv(k) → ReLU → [LRN?] → [MaxPool?] → out


GUI에서는 한 개 블록으로 보이고, Expand에서 내부 원자 연산을 설명

Export 시:

단일 파일: 헤더에 class SEBlock/ResidualBlock/VGGBlock 포함

ZIP 패키지: blocks.py로 분리

검증 규칙 요약

토폴로지: DAG(사이클 금지), 미연결 노드 경고

랭크/레이아웃

1D: (B,C,T), 2D: (B,C,H,W)

RNN 입력: (B,T,H) → 필요 시 Permute 제안

머지

Concat: rank 동일, 병합축 외 모든 축 동일

Add: rank/shape 완전 동일

파라미터: kernel_size>0, out_channels>0, reduction≥1…

컴포지트: 외부 인터페이스는 (MVP) 입력1/출력1

아키텍처
┌─────────────────────────────── Frontend ───────────────────────────────┐
│ React (React Flow + CodeMirror) / Streamlit demo                       │
│  - 드래그·드롭, 파라미터 편집, 내부보기(Expand), 코드 미리보기/다운로드     │
└───────────────▲─────────────────────────────┬───────────────────────────┘
                │  Graph JSON                 │  .py / .zip
     /registry  │  /validate                  │  /codegen  /export/zip
                │                              ▼
┌───────────────┴────────────────────────────────────────────────────────┐
│ FastAPI Backend                                                        │
│  - NODE_SPECS 검증(토폴로지/랭크/머지/파라미터)                         │
│  - Macro Resolver(컴포지트 해석) → Codegen Adapter                      │
│  - Codegen 호출 → model.py / blocks.py / train_dummy.py                 │
└───────────────┬────────────────────────────────────────────────────────┘
                ▼
        Python Codegen Library
        - render_model_py()
        - render_blocks_py()   # SE/Residual/VGG
        - package_zip()

템플릿 예시
VGG-16(간략) JSON (2D)
{
  "nodes": [
    {"id":"inp","type":"Input","params":{}},

    {"id":"b1","type":"VGGBlock","params":{"c1":64,"c2":64,"k":3,"pool":true}},
    {"id":"b2","type":"VGGBlock","params":{"c1":128,"c2":128,"k":3,"pool":true}},
    {"id":"b3","type":"VGGBlock","params":{"c1":256,"c2":256,"k":3,"pool":true}},
    {"id":"b4","type":"VGGBlock","params":{"c1":512,"c2":512,"k":3,"pool":true}},
    {"id":"b5","type":"VGGBlock","params":{"c1":512,"c2":512,"k":3,"pool":true}},

    {"id":"flat","type":"Flatten","params":{}},
    {"id":"fc1","type":"Linear","params":{"out_features":4096}},
    {"id":"relu1","type":"ReLU","params":{}},
    {"id":"fc2","type":"Linear","params":{"out_features":4096}},
    {"id":"relu2","type":"ReLU","params":{}},
    {"id":"fc3","type":"Linear","params":{"out_features":1000}}
  ],
  "edges": [
    ["inp","b1"],["b1","b2"],["b2","b3"],["b3","b4"],["b4","b5"],
    ["b5","flat"],["flat","fc1"],["fc1","relu1"],["relu1","fc2"],["fc2","relu2"],["relu2","fc3"]
  ],
  "inputs":["inp"],"outputs":["fc3"]
}

1D(데모) JSON
{
  "nodes": [
    {"id":"inp","type":"Input","params":{}},
    {"id":"conv","type":"Conv1d","params":{"out_channels":64,"kernel_size":5,"padding":"same"}},
    {"id":"relu","type":"ReLU","params":{}},
    {"id":"perm","type":"Permute_BCT_to_BTH","params":{}},
    {"id":"gru","type":"GRUBlock","params":{"hidden_size":128,"out":"last"}},
    {"id":"fc","type":"Linear","params":{"out_features":18}}
  ],
  "edges":[["inp","conv"],["conv","relu"],["relu","perm"],["perm","gru"],["gru","fc"]],
  "inputs":["inp"],"outputs":["fc"]
}

로드맵
M1 — Codegen & 검증기 (주 1)

원자 블록(1D/2D), Concat/Add, Permute/Flatten

NODE_SPECS 검증(토폴로지/머지/파라미터)

단일 파일 Export + train_dummy

M2 — 컴포지트 & 템플릿 (주 2)

SEBlock/ResidualBlock/VGGBlock 매핑 + Expand 패널

템플릿 3종(1D HybridCMI, 2D VGG-16/ResNet-18) 스모크 통과

M3 — FastAPI + React (주 3)

/registry /validate /codegen /export/zip

React Flow + CodeMirror + ZIP Export

M4 — 품질 (주 4)

유닛/통합/E2E 테스트, README/Quickstart, 60초 데모 영상

리포 구조
torchcanvas/
  backend/
    server.py            # FastAPI
    validator.py         # NODE_SPECS 검증
    codegen_adapter.py   # Graph→codegen bridge
  codegen/
    model_codegen.py     # render_model_py()
    blocks_codegen.py    # render_blocks_py() (SE/Residual/VGG)
    package.py           # zip 스트리밍
  frontend/
    app.py               # Streamlit 미니 GUI (현행)
  templates/
    vgg16.json
    resnet18.json
    hybridcmi.json
  tools/
    train_dummy.py
  README.md

테스트 전략

단위(Unit): 각 노드 입/출력 rank·shape·param 범위 체크

그래프 통합: 직렬/브랜치/Skip/머지 시나리오, 1D+2D, RNN 경로

회귀(snapshot): 템플릿 JSON → 코드 문자열 diff, forward() 결과 shape

E2E: /validate→/codegen→Export 코드 실행→train_dummy 1 epoch

라이선스

TBD (Apache-2.0 또는 MIT 권장)

한 줄 요약:

“그리듯 설계하고, 즉시 PyTorch 코드로 가져가 학습하자.”