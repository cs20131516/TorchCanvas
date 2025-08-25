# app.py — TorchCanvas Mini GUI (Graph → PyTorch 코드 미리보기/다운로드)
# 실행: streamlit run app.py
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="TorchCanvas — Code Preview", layout="wide")

# ---------- 개선된 노드 타입 스펙 (카테고리별) ----------
NODE_SPECS = {
    # 입력/출력
    "Input": {
        "params": {},
        "category": "io",
        "color": "#28a745",
        "icon": "📥",
        "description": "모델 입력"
    },
    "Output": {
        "params": {},
        "category": "io", 
        "color": "#17a2b8",
        "icon": "📤",
        "description": "모델 출력"
    },
    
    # 컨볼루션 레이어
    "Conv1d": {
        "params": {"out_channels": int, "kernel_size": int, "stride": (int, 1), "padding": ("same_or_int", "same")},
        "category": "conv",
        "color": "#ffc107",
        "icon": "🔲",
        "description": "1D 컨볼루션"
    },
    "Conv2d": {
        "params": {"out_channels": int, "kernel_size": int, "stride": (int, 1), "padding": ("same_or_int", "same")},
        "category": "conv",
        "color": "#ffc107", 
        "icon": "🔲",
        "description": "2D 컨볼루션"
    },
    
    # 정규화 레이어
    "BatchNorm1d": {
        "params": {"num_features": (int, 0)},
        "category": "norm",
        "color": "#6f42c1",
        "icon": "📊",
        "description": "1D 배치 정규화"
    },
    "BatchNorm2d": {
        "params": {"num_features": (int, 0)},
        "category": "norm",
        "color": "#6f42c1",
        "icon": "📊", 
        "description": "2D 배치 정규화"
    },
    
    # 활성화 함수
    "ReLU": {
        "params": {},
        "category": "activation",
        "color": "#007bff",
        "icon": "⚡",
        "description": "ReLU 활성화"
    },
    "Sigmoid": {
        "params": {},
        "category": "activation", 
        "color": "#007bff",
        "icon": "📈",
        "description": "Sigmoid 활성화"
    },
    "Tanh": {
        "params": {},
        "category": "activation",
        "color": "#007bff", 
        "icon": "📈",
        "description": "Tanh 활성화"
    },
    
    # 풀링 레이어
    "MaxPool1d": {
        "params": {"kernel_size": (int, 2), "stride": (int, None)},
        "category": "pool",
        "color": "#fd7e14",
        "icon": "🔽",
        "description": "1D 최대 풀링"
    },
    "MaxPool2d": {
        "params": {"kernel_size": (int, 2), "stride": (int, None)},
        "category": "pool",
        "color": "#fd7e14",
        "icon": "🔽",
        "description": "2D 최대 풀링"
    },
    "AvgPool2d": {
        "params": {"kernel_size": (int, 2), "stride": (int, None)},
        "category": "pool",
        "color": "#fd7e14",
        "icon": "🔽",
        "description": "2D 평균 풀링"
    },
    
    # 완전연결 레이어
    "Linear": {
        "params": {"out_features": int, "bias": (bool, True)},
        "category": "fc",
        "color": "#e83e8c",
        "icon": "🔗",
        "description": "완전연결 레이어"
    },
    
    # 유틸리티
    "Dropout": {
        "params": {"p": (float, 0.5)},
        "category": "util",
        "color": "#6c757d",
        "icon": "🎲",
        "description": "드롭아웃"
    },
    "Flatten": {
        "params": {"start_dim": (int, 1), "end_dim": (int, -1)},
        "category": "util",
        "color": "#6c757d",
        "icon": "📏",
        "description": "텐서 평탄화"
    },
    
    # 결합 연산
    "Concat": {
        "params": {"dim": (int, 1)},
        "category": "combine",
        "color": "#20c997",
        "icon": "🔗",
        "description": "텐서 연결"
    },
    "Add": {
        "params": {},
        "category": "combine",
        "color": "#20c997",
        "icon": "➕",
        "description": "텐서 덧셈"
    },
    
    # 순열 연산
    "Permute_BCT_to_BTH": {
        "params": {},
        "category": "permute",
        "color": "#6f42c1",
        "icon": "🔄",
        "description": "BCT → BTH 변환"
    },
    "Permute_BTH_to_BCT": {
        "params": {},
        "category": "permute", 
        "color": "#6f42c1",
        "icon": "🔄",
        "description": "BTH → BCT 변환"
    },
    
    # 순환 신경망
    "GRUBlock": {
        "params": {
            "hidden_size": int, "num_layers": (int, 1), "bidirectional": (bool, True),
            "out": (["last","mean","max","seq"], "last")
        },
        "category": "rnn",
        "color": "#dc3545",
        "icon": "🔄",
        "description": "GRU 블록"
    },
    
    # 컴포지트 블록들
    "SEBlock": {
        "params": {"reduction": (int, 8)},
        "category": "composite",
        "color": "#28a745",
        "icon": "🎯",
        "description": "Squeeze-and-Excitation"
    },
    "ResidualBlock": {
        "params": {"out_channels": int, "kernel_size": (int, 3), "stride": (int, 1)},
        "category": "composite",
        "color": "#28a745",
        "icon": "⏭️",
        "description": "잔차 블록"
    },
    "VGGBlock": {
        "params": {
            "c1": int, "c2": int, "kernel_size": (int, 3), 
            "use_lrn": (bool, False), "pool": (bool, True)
        },
        "category": "composite",
        "color": "#28a745",
        "icon": "🏗️",
        "description": "VGG 스타일 블록"
    },
}

# 카테고리별 그룹핑
CATEGORIES = {
    "io": "입력/출력",
    "conv": "컨볼루션", 
    "norm": "정규화",
    "activation": "활성화",
    "pool": "풀링",
    "fc": "완전연결",
    "util": "유틸리티",
    "combine": "결합",
    "permute": "순열",
    "rnn": "순환신경망",
    "composite": "컴포지트"
}

# ---------- 세션 상태 초기화 ----------
def _init_state():
    ss = st.session_state
    ss.setdefault("nodes", [])    # type: ignore[assignment]  ← 필요시
    ss.setdefault("edges", [])
    ss.setdefault("inputs", [])
    ss.setdefault("outputs", [])
    ss.setdefault("tensor_shapes", {})  # 노드별 텐서 형태 저장
_init_state()

# ---------- 텐서 형태 추정 함수 ----------
def estimate_tensor_shape(node_id: str, node_type: str, params: dict, input_shapes: List[tuple]) -> tuple:
    """노드의 출력 텐서 형태를 추정"""
    if not input_shapes:
        return (1, 3, 224, 224)  # 기본 입력 형태
    
    input_shape = input_shapes[0]  # 첫 번째 입력 사용
    
    if node_type == "Input":
        return input_shape
    
    elif node_type == "Conv2d":
        B, C, H, W = input_shape
        out_channels = params.get("out_channels", 64)
        kernel_size = params.get("kernel_size", 3)
        stride = params.get("stride", 1)
        padding = params.get("padding", "same")
        
        if padding == "same":
            pad = kernel_size // 2
        else:
            pad = int(padding)
        
        H_out = (H + 2 * pad - kernel_size) // stride + 1
        W_out = (W + 2 * pad - kernel_size) // stride + 1
        return (B, out_channels, H_out, W_out)
    
    elif node_type == "MaxPool2d":
        B, C, H, W = input_shape
        kernel_size = params.get("kernel_size", 2)
        stride = params.get("stride", kernel_size)
        
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        return (B, C, H_out, W_out)
    
    elif node_type == "ReLU":
        return input_shape  # 형태 유지
    
    elif node_type == "LRN":
        return input_shape  # 형태 유지
    
    elif node_type == "Flatten":
        B, C, H, W = input_shape
        return (B, C * H * W)
    
    elif node_type == "Linear":
        B = input_shape[0]
        out_features = params.get("out_features", 1000)
        return (B, out_features)
    
    # 기본적으로 입력 형태 유지
    return input_shape

# ---------- 개선된 시각적 네트워크 다이어그램 생성 ----------
def create_network_diagram(nodes: List[dict], edges: List[List[str]], tensor_shapes: Dict[str, tuple]) -> str:
    """HTML/CSS로 네트워크 다이어그램 생성"""
    
    # 의미없는 연결 필터링 (Input → Output, 같은 타입 간 연결 등)
    def is_meaningful_edge(src_id: str, dst_id: str) -> bool:
        src_node = next((n for n in nodes if n["id"] == src_id), None)
        dst_node = next((n for n in nodes if n["id"] == dst_id), None)
        
        if not src_node or not dst_node:
            return False
        
        # Input → Output 연결 제외
        if src_node["type"] == "Input" and dst_node["type"] == "Output":
            return False
        
        # 같은 타입 간 직접 연결 제외 (예: Conv1d → Conv1d)
        if src_node["type"] == dst_node["type"] and src_node["type"] in ["Conv1d", "Conv2d", "BatchNorm1d", "BatchNorm2d"]:
            return False
        
        return True
    
    filtered_edges = [edge for edge in edges if is_meaningful_edge(edge[0], edge[1])]
    
    html = """
    <style>
    .network-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .layers-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
        position: relative;
        min-height: 400px;
    }
    .layer {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        position: relative;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border-left: 5px solid;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .layer:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
    }
    .layer-header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    .layer-icon {
        font-size: 24px;
        margin-right: 10px;
    }
    .layer-name {
        font-weight: bold;
        font-size: 16px;
        margin: 0;
    }
    .layer-id {
        font-size: 12px;
        color: #6c757d;
        margin-top: 5px;
    }
    .tensor-shape {
        font-family: 'Courier New', monospace;
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        padding: 8px 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 13px;
        font-weight: bold;
        border: 1px solid #dee2e6;
    }
    .status-check {
        position: absolute;
        top: 15px;
        right: 15px;
        color: #28a745;
        font-size: 20px;
        background: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }
    .params {
        font-size: 12px;
        color: #6c757d;
        margin-top: 10px;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 6px;
        border-left: 3px solid #dee2e6;
    }
    .category-badge {
        position: absolute;
        top: 15px;
        left: 15px;
        font-size: 10px;
        padding: 4px 8px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-transform: uppercase;
    }
    .connection-info {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .connection-count {
        font-size: 18px;
        font-weight: bold;
        color: #667eea;
    }
    .connection-details {
        font-size: 14px;
        color: #6c757d;
        margin-top: 5px;
    }
    .warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        color: #856404;
    }
    .connection-lines {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }
    .connection-line {
        stroke: #667eea;
        stroke-width: 2;
        fill: none;
        opacity: 0.7;
        marker-end: url(#arrowhead);
    }
    .layer {
        position: relative;
        z-index: 2;
    }
    </style>
    <div class="network-container">
    """
    
    # 연결 정보 표시
    meaningful_count = len(filtered_edges)
    total_count = len(edges)
    removed_count = total_count - meaningful_count
    
    html += f"""
    <div class="connection-info">
        <div class="connection-count">🔗 연결 정보</div>
        <div class="connection-details">
            유효한 연결: {meaningful_count}개 | 
            필터링된 연결: {removed_count}개 | 
            총 연결: {total_count}개
        </div>
    </div>
    """
    
    if removed_count > 0:
        html += f"""
        <div class="warning">
            ⚠️ {removed_count}개의 의미없는 연결이 자동으로 필터링되었습니다.
            (Input→Output, 같은 타입 간 직접 연결 등)
        </div>
        """
    
    # 노드들을 그리드 형태로 배치
    html += '<div class="layers-grid" style="position: relative;">'
    
    # SVG 연결선을 위한 컨테이너
    html += '''
    <svg class="connection-lines" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                    refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#667eea" />
            </marker>
        </defs>
    '''
    
    # 노드 위치 추적을 위한 딕셔너리
    node_positions = {}
    
    for i, node in enumerate(nodes):
        node_id = node["id"]
        node_type = node["type"]
        params = node.get("params", {})
        
        # 텐서 형태 가져오기
        input_shapes = []
        for src, dst in filtered_edges:
            if dst == node_id and src in tensor_shapes:
                input_shapes.append(tensor_shapes[src])
        
        if node_id in tensor_shapes:
            output_shape = tensor_shapes[node_id]
        else:
            output_shape = estimate_tensor_shape(node_id, node_type, params, input_shapes)
            st.session_state.tensor_shapes[node_id] = output_shape
        
        # 파라미터 문자열 생성
        param_str = ""
        if params:
            param_items = []
            for k, v in params.items():
                if isinstance(v, (int, float, bool)):
                    param_items.append(f"{k}={v}")
                elif isinstance(v, str):
                    param_items.append(f"{k}='{v}'")
            param_str = ", ".join(param_items)
        
        # 카테고리 색상 가져오기
        category_color = NODE_SPECS[node_type]["color"]
        category_name = NODE_SPECS[node_type]["category"]
        icon = NODE_SPECS[node_type]["icon"]
        
        # 연결 정보 추가
        incoming = [src for src, dst in filtered_edges if dst == node_id]
        outgoing = [dst for src, dst in filtered_edges if src == node_id]
        connection_info = f"입력: {len(incoming)}개, 출력: {len(outgoing)}개"
        
        # 노드 위치 계산 (그리드 위치)
        grid_col = i % 3  # 3열 그리드
        grid_row = i // 3
        x_pos = grid_col * 320 + 160  # 300px 너비 + 20px gap
        y_pos = grid_row * 200 + 100  # 150px 높이 + 20px gap
        
        node_positions[node_id] = (x_pos, y_pos)
        
        html += f"""
        <div class="layer" style="border-left-color: {category_color};" data-node-id="{node_id}">
            <div class="category-badge" style="background-color: {category_color};">{category_name}</div>
            <div class="status-check">✓</div>
            <div class="layer-header">
                <div class="layer-icon">{icon}</div>
                <div class="layer-name">{node_type}</div>
            </div>
            <div class="layer-id">ID: {node_id}</div>
            <div class="tensor-shape">[{', '.join(map(str, output_shape))}]</div>
            <div class="connection-details" style="margin-top: 10px; font-size: 11px;">{connection_info}</div>
            {f'<div class="params">{param_str}</div>' if param_str else ''}
        </div>
        """
    
    # 연결선 그리기
    for src, dst in filtered_edges:
        if src in node_positions and dst in node_positions:
            x1, y1 = node_positions[src]
            x2, y2 = node_positions[dst]
            
            # 곡선 연결선 (베지어 곡선)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            html += f'''
            <path class="connection-line" 
                  d="M {x1} {y1} Q {mid_x} {y1} {mid_x} {mid_y} T {x2} {y2}"
                  marker-end="url(#arrowhead)" />
            '''
    
    html += "</svg></div></div>"
    return html

# ---------- 코드 생성기 ----------
def export_graph_to_python(graph: dict, class_name: str="ExportedModel") -> str:
    nodes = graph["nodes"]; edges = graph["edges"]
    inputs = graph["inputs"]; outputs = graph["outputs"]
    node_types = {n["id"]: n["type"] for n in nodes}
    node_params = {n["id"]: n.get("params", {}) for n in nodes}
    in_edges = {n["id"]: [] for n in nodes}
    for src, dst in edges:
        in_edges[dst].append(src)
    # topo sort
    indeg = {n["id"]: 0 for n in nodes}
    adj = {}
    for src, dst in edges:
        indeg[dst] += 1
        adj.setdefault(src, []).append(dst)
    q = [nid for nid, d in indeg.items() if d == 0]
    order = []
    while q:
        u = q.pop()
        order.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(nodes):
        raise ValueError("Graph has cycle or disconnected parts.")

    # __init__
    init_lines = []
    for nid in order:
        typ = node_types[nid]
        if typ == "Input":
            continue
        p = node_params[nid]
        if typ == "Conv1d":
            pad = p.get("padding", "same")
            pad_expr = f"{p.get('kernel_size')} // 2" if pad == "same" else str(int(pad))
            init_lines.append(
                f"self.{nid} = nn.LazyConv1d({p['out_channels']}, {p['kernel_size']}, stride={p.get('stride',1)}, padding={pad_expr}, bias=True)"
            )
        elif typ == "Conv2d":
            pad = p.get("padding", "same")
            pad_expr = f"{p.get('kernel_size')} // 2" if pad == "same" else str(int(pad))
            init_lines.append(
                f"self.{nid} = nn.LazyConv2d({p['out_channels']}, {p['kernel_size']}, stride={p.get('stride',1)}, padding={pad_expr}, bias=True)"
            )
        elif typ == "BatchNorm1d":
            num = p.get("num_features", 0)
            if num in (None, 0):
                init_lines.append(f"self.{nid} = nn.LazyBatchNorm1d()")
            else:
                init_lines.append(f"self.{nid} = nn.BatchNorm1d({num})")
        elif typ == "BatchNorm2d":
            num = p.get("num_features", 0)
            if num in (None, 0):
                init_lines.append(f"self.{nid} = nn.LazyBatchNorm2d()")
            else:
                init_lines.append(f"self.{nid} = nn.BatchNorm2d({num})")
        elif typ == "Linear":
            init_lines.append(f"self.{nid} = nn.LazyLinear({p['out_features']}, bias={p.get('bias', True)})")
        elif typ == "ReLU":
            init_lines.append(f"self.{nid} = nn.ReLU(inplace=True)")
        elif typ == "Sigmoid":
            init_lines.append(f"self.{nid} = nn.Sigmoid()")
        elif typ == "Tanh":
            init_lines.append(f"self.{nid} = nn.Tanh()")
        elif typ == "Dropout":
            init_lines.append(f"self.{nid} = nn.Dropout({p.get('p', 0.5)})")
        elif typ == "MaxPool1d":
            ks = p.get("kernel_size", 2); stv = p.get("stride", None)
            args = [f"kernel_size={ks}"]; 
            if stv is not None: args.append(f"stride={stv}")
            init_lines.append(f"self.{nid} = nn.MaxPool1d({', '.join(args)})")
        elif typ == "MaxPool2d":
            ks = p.get("kernel_size", 2); stv = p.get("stride", None)
            args = [f"kernel_size={ks}"]; 
            if stv is not None: args.append(f"stride={stv}")
            init_lines.append(f"self.{nid} = nn.MaxPool2d({', '.join(args)})")
        elif typ == "Flatten":
            start_dim = p.get("start_dim", 1)
            end_dim = p.get("end_dim", -1)
            init_lines.append(f"self.{nid} = nn.Flatten(start_dim={start_dim}, end_dim={end_dim})")
        elif typ == "GRUBlock":
            hs = p["hidden_size"]; nl = p.get("num_layers", 1); bd = p.get("bidirectional", True)
            outm = repr(p.get("out", "last"))
            init_lines.append(f"self.{nid} = GRUBlock(hidden_size={hs}, num_layers={nl}, bidirectional={bd}, out={outm})")
        elif typ == "SEBlock":
            reduction = p.get("reduction", 8)
            init_lines.append(f"self.{nid} = SEBlock(reduction={reduction})")
        elif typ == "ResidualBlock":
            out_channels = p["out_channels"]
            kernel_size = p.get("kernel_size", 3)
            stride = p.get("stride", 1)
            init_lines.append(f"self.{nid} = ResidualBlock(out_channels={out_channels}, kernel_size={kernel_size}, stride={stride})")
        elif typ == "VGGBlock":
            c1 = p["c1"]; c2 = p["c2"]; k = p.get("kernel_size", 3)
            use_lrn = p.get("use_lrn", False); pool = p.get("pool", True)
            init_lines.append(f"self.{nid} = VGGBlock(c1={c1}, c2={c2}, kernel_size={k}, use_lrn={use_lrn}, pool={pool})")
        elif typ in ("Concat","Add","Permute_BCT_to_BTH","Permute_BTH_to_BCT"):
            init_lines.append(f"# {typ} '{nid}' has no parameters")
        else:
            init_lines.append(f"# TODO: Unsupported node type '{typ}'")

    init_body = "\n        ".join(init_lines)

    # forward()
    fwd = []
    fwd.append("cache = {}")
    for inp in inputs:
        fwd.append(f"cache['{inp}'] = inputs['{inp}']")
    for nid in order:
        typ = node_types[nid]
        if typ == "Input":
            continue
        srcs = in_edges[nid]
        if typ == "Concat":
            dim = node_params[nid].get("dim", 1)
            src_list = ", ".join([f"cache['{s}']" for s in srcs])
            fwd.append(f"cache['{nid}'] = torch.cat([{src_list}], dim={dim})")
        elif typ == "Add":
            if len(srcs) != 2:
                raise ValueError(f"Add node '{nid}' requires exactly 2 inputs, got {len(srcs)}")
            fwd.append(f"cache['{nid}'] = cache['{srcs[0]}'] + cache['{srcs[1]}']")
        elif typ == "Permute_BCT_to_BTH":
            fwd.append(f"cache['{nid}'] = cache['{srcs[0]}'].transpose(1, 2).contiguous()")
        elif typ == "Permute_BTH_to_BCT":
            fwd.append(f"cache['{nid}'] = cache['{srcs[0]}'].transpose(1, 2).contiguous()")
        elif typ in ("Conv1d","Conv2d","BatchNorm1d","BatchNorm2d","Linear","ReLU","Sigmoid","Tanh","Dropout","MaxPool1d","MaxPool2d","AvgPool2d","Flatten","GRUBlock","SEBlock","ResidualBlock","VGGBlock"):
            fwd.append(f"cache['{nid}'] = self.{nid}(cache['{srcs[0]}'])")
        else:
            fwd.append(f"# TODO: forward for '{typ}'")
    if len(outputs) == 1:
        fwd.append(f"return cache['{outputs[0]}']")
    else:
        pairs = ", ".join([f"'{k}': cache['{k}']" for k in outputs])
        fwd.append(f"return {{ {pairs} }}")
    fwd_body = "\n        ".join(fwd)

    header = '''# Auto-generated by TorchCanvas Code Export
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUBlock(nn.Module):
    """
    GRU + output normalization
    - input: (B, T, H_in)
    - out: 'last' | 'mean' | 'max' | 'seq'
    """
    def __init__(self, hidden_size:int, num_layers:int=1, bidirectional:bool=True, out:str="last"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.out_mode = out
        self.gru = None  # lazy init at first forward

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"GRUBlock expects 3D (B,T,H), got {tuple(x.shape)}")
        if self.gru is None:
            self.gru = nn.GRU(
                input_size=x.size(-1),
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            ).to(x.device, x.dtype)
        y, _ = self.gru(x)
        if self.out_mode == "seq":
            return y
        if self.out_mode == "last":
            return y[:, -1, :]
        if self.out_mode == "mean":
            return y.mean(dim=1)
        if self.out_mode == "max":
            return y.max(dim=1).values
        raise ValueError(f"Unknown out mode: {self.out_mode}")

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    - input: (B, C, H, W) or (B, C, T)
    - reduction: channel reduction ratio
    """
    def __init__(self, reduction: int = 8):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = None
        self.fc1 = None
        self.fc2 = None
        
    def forward(self, x):
        if self.avg_pool is None:
            # Lazy initialization based on input shape
            if x.ndim == 4:  # (B, C, H, W)
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
            else:  # (B, C, T)
                self.avg_pool = nn.AdaptiveAvgPool1d(1)
            
            channels = x.size(1)
            self.fc1 = nn.Linear(channels, channels // self.reduction).to(x.device, x.dtype)
            self.fc2 = nn.Linear(channels // self.reduction, channels).to(x.device, x.dtype)
        
        # Squeeze
        if x.ndim == 4:
            y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        else:
            y = self.avg_pool(x).squeeze(-1)  # (B, C)
        
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        
        # Scale
        if x.ndim == 4:
            return x * y.unsqueeze(-1).unsqueeze(-1)
        else:
            return x * y.unsqueeze(-1)

class ResidualBlock(nn.Module):
    """
    Residual Block with optional projection
    - input: (B, C, H, W) or (B, C, T)
    - out_channels: output channels (if different from input, uses 1x1 projection)
    """
    def __init__(self, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = None
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        self.projection = None
        
    def forward(self, x):
        if self.conv1 is None:
            # Lazy initialization
            in_channels = x.size(1)
            padding = self.kernel_size // 2
            
            if x.ndim == 4:  # 2D
                self.conv1 = nn.Conv2d(in_channels, self.out_channels, self.kernel_size, 
                                     stride=self.stride, padding=padding, bias=False).to(x.device, x.dtype)
                self.bn1 = nn.BatchNorm2d(self.out_channels).to(x.device, x.dtype)
                self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size,
                                     padding=padding, bias=False).to(x.device, x.dtype)
                self.bn2 = nn.BatchNorm2d(self.out_channels).to(x.device, x.dtype)
                
                if in_channels != self.out_channels or self.stride != 1:
                    self.projection = nn.Conv2d(in_channels, self.out_channels, 1, 
                                              stride=self.stride, bias=False).to(x.device, x.dtype)
            else:  # 1D
                self.conv1 = nn.Conv1d(in_channels, self.out_channels, self.kernel_size,
                                     stride=self.stride, padding=padding, bias=False).to(x.device, x.dtype)
                self.bn1 = nn.BatchNorm1d(self.out_channels).to(x.device, x.dtype)
                self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size,
                                     padding=padding, bias=False).to(x.device, x.dtype)
                self.bn2 = nn.BatchNorm1d(self.out_channels).to(x.device, x.dtype)
                
                if in_channels != self.out_channels or self.stride != 1:
                    self.projection = nn.Conv1d(in_channels, self.out_channels, 1,
                                              stride=self.stride, bias=False).to(x.device, x.dtype)
        
        identity = x
        if self.projection is not None:
            identity = self.projection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out

class VGGBlock(nn.Module):
    """
    VGG-style block: Conv-ReLU-Conv-ReLU-(Pool)
    - input: (B, C, H, W) or (B, C, T)
    - c1, c2: channel counts for two conv layers
    """
    def __init__(self, c1: int, c2: int, kernel_size: int = 3, use_lrn: bool = False, pool: bool = True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.kernel_size = kernel_size
        self.use_lrn = use_lrn
        self.pool = pool
        self.conv1 = None
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        self.pool_layer = None
        
    def forward(self, x):
        if self.conv1 is None:
            # Lazy initialization
            in_channels = x.size(1)
            padding = self.kernel_size // 2
            
            if x.ndim == 4:  # 2D
                self.conv1 = nn.Conv2d(in_channels, self.c1, self.kernel_size, padding=padding, bias=False).to(x.device, x.dtype)
                self.bn1 = nn.BatchNorm2d(self.c1).to(x.device, x.dtype)
                self.conv2 = nn.Conv2d(self.c1, self.c2, self.kernel_size, padding=padding, bias=False).to(x.device, x.dtype)
                self.bn2 = nn.BatchNorm2d(self.c2).to(x.device, x.dtype)
                if self.pool:
                    self.pool_layer = nn.MaxPool2d(2, 2).to(x.device, x.dtype)
            else:  # 1D
                self.conv1 = nn.Conv1d(in_channels, self.c1, self.kernel_size, padding=padding, bias=False).to(x.device, x.dtype)
                self.bn1 = nn.BatchNorm1d(self.c1).to(x.device, x.dtype)
                self.conv2 = nn.Conv1d(self.c1, self.c2, self.kernel_size, padding=padding, bias=False).to(x.device, x.dtype)
                self.bn2 = nn.BatchNorm1d(self.c2).to(x.device, x.dtype)
                if self.pool:
                    self.pool_layer = nn.MaxPool1d(2, 2).to(x.device, x.dtype)
        
        x = F.relu(self.bn1(self.conv1(x)))
        if self.use_lrn and x.ndim == 4:
            x = F.local_response_normalization(x, size=5)
        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_lrn and x.ndim == 4:
            x = F.local_response_normalization(x, size=5)
        if self.pool and self.pool_layer is not None:
            x = self.pool_layer(x)
        
        return x
'''
    cls = f"""
class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()
        {init_body}

    def forward(self, inputs: dict):
        {fwd_body}
"""
    main = """
if __name__ == "__main__":
    # quick smoke test
    model = ExportedModel()
    x = torch.randn(4, 3, 224, 224)  # (B,C,H,W) for 2D or (B,C,T) for 1D
    out = model({"inp": x})
    print("out.shape =", out.shape)
"""
    return header + cls + main

# ---------- 개선된 사이드바: 시각적 팔레트 ----------
st.sidebar.title("🎨 TorchCanvas — Palette")

# 카테고리별 팔레트
selected_category = st.sidebar.selectbox(
    "카테고리 선택",
    list(CATEGORIES.keys()),
    format_func=lambda x: CATEGORIES[x],
    index=0
)

# 선택된 카테고리의 블록들 표시
st.sidebar.subheader(f"📂 {CATEGORIES[selected_category]}")

# 카테고리별 블록들을 시각적으로 표시
category_blocks = [name for name, spec in NODE_SPECS.items() if spec["category"] == selected_category]

for block_name in category_blocks:
    block_spec = NODE_SPECS[block_name]
    
    # 블록 카드 스타일
    with st.sidebar.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<div style='text-align: center; font-size: 20px;'>{block_spec['icon']}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{block_name}**")
            st.caption(block_spec['description'])
        
        # 블록 추가 버튼
        if st.button(f"➕ {block_name} 추가", key=f"add_{block_name}"):
            # 자동으로 노드 ID 생성
            base_name = block_name.lower()
            existing_ids = [n["id"] for n in st.session_state.nodes]
            counter = 1
            while f"{base_name}{counter}" in existing_ids:
                counter += 1
            nid = f"{base_name}{counter}"
            
            # 기본 파라미터로 노드 추가
            params = {}
            for k, spec in block_spec["params"].items():
                if isinstance(spec, tuple):
                    typ, default = spec
                    if typ == int:
                        params[k] = default if default is not None else 1
                    elif typ == float:
                        params[k] = float(default) if default is not None else 0.5
                    elif typ == bool:
                        params[k] = bool(default) if default is not None else False
                    else:
                        params[k] = default
                elif spec == int:
                    params[k] = 64  # 기본값
                elif spec == float:
                    params[k] = 0.5
                elif spec == bool:
                    params[k] = True
                elif spec == "same_or_int":
                    params[k] = "same"
                else:
                    params[k] = spec
            
            st.session_state.nodes.append({"id": nid, "type": block_name, "params": params})
            st.rerun()
        
        st.sidebar.divider()

# 고급 설정 (접을 수 있는 섹션)
with st.sidebar.expander("⚙️ 고급 설정"):
    st.subheader("수동 노드 추가")
    with st.form("manual_add_node_form"):
        nid = st.text_input("노드 ID", placeholder="예: conv1")
        ntype = st.selectbox("노드 타입", list(NODE_SPECS.keys()), index=0)
        params_spec = NODE_SPECS[ntype]["params"]
        params = {}
        for k, spec in params_spec.items():
            if spec == int:
                params[k] = st.number_input(k, value=1, step=1)
            elif spec == float:
                params[k] = st.number_input(k, value=0.5, step=0.1, format="%.4f")
            elif spec == bool:
                params[k] = st.checkbox(k, value=False)
            elif spec == "same_or_int":
                mode = st.selectbox(k, ["same", "int"], index=0)
                params[k] = "same" if mode == "same" else int(st.number_input(f"{k} (int)", value=1, step=1))
            elif isinstance(spec, tuple):
                typ, default = spec
                if typ == int:
                    val = st.number_input(k, value=default if default is not None else 0, step=1)
                    params[k] = int(val) if default is not None else (None if val==0 else int(val))
                elif typ == float:
                    params[k] = float(st.number_input(k, value=float(default)))
                elif typ == bool:
                    params[k] = st.checkbox(k, value=bool(default))
                else:
                    params[k] = default
            elif isinstance(spec, list):  # enum
                default = NODE_SPECS[ntype]["params"][k][1] if isinstance(NODE_SPECS[ntype]["params"][k], tuple) else spec[0]
                params[k] = st.selectbox(k, spec, index=spec.index(default) if default in spec else 0)
            else:
                params[k] = st.text_input(k, value=str(spec))
        add = st.form_submit_button("수동 추가")
        if add:
            assert nid, "노드 ID는 필수"
            st.session_state.nodes.append({"id": nid, "type": ntype, "params": params})
            st.rerun()

# ---------- 사이드바: 엣지/입출력 ----------
st.sidebar.subheader("Edges")
if st.session_state.nodes:
    opts = [n["id"] for n in st.session_state.nodes]
    with st.sidebar.form("add_edge_form"):
        src = st.selectbox("src", opts)
        dst = st.selectbox("dst", opts)
        add_e = st.form_submit_button("엣지 추가")
        if add_e:
            st.session_state.edges.append([src, dst]); st.rerun()

st.sidebar.divider()
st.sidebar.subheader("입력/출력 설정")
all_ids = [n["id"] for n in st.session_state.nodes]
st.session_state.inputs = st.sidebar.multiselect("inputs", all_ids, default=st.session_state.inputs)
st.session_state.outputs = st.sidebar.multiselect("outputs", all_ids, default=st.session_state.outputs)

st.sidebar.divider()
if st.sidebar.button("그래프 초기화"):
    st.session_state.nodes.clear(); st.session_state.edges.clear()
    st.session_state.inputs.clear(); st.session_state.outputs.clear()
    st.session_state.tensor_shapes.clear()
    st.rerun()

# ---------- 메인: 탭 기반 인터페이스 ----------
st.title("TorchCanvas — 시각적 신경망 디자이너")

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["🎨 네트워크 시각화", "⚙️ 코드 생성", "📊 상세 정보", "📋 템플릿"])

with tab1:
    # 네트워크 다이어그램 표시 (더 큰 크기)
    if st.session_state.nodes:
        st.subheader("🔍 네트워크 아키텍처 시각화")
        
        # 네트워크 통계 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 레이어", len(st.session_state.nodes))
        with col2:
            st.metric("연결 수", len(st.session_state.edges))
        with col3:
            st.metric("입력", len(st.session_state.inputs))
        with col4:
            st.metric("출력", len(st.session_state.outputs))
        
        # 더 큰 다이어그램 (개선된 버전 사용)
        diagram_html = create_network_diagram(
            st.session_state.nodes, 
            st.session_state.edges, 
            st.session_state.tensor_shapes
        )
        st.components.v1.html(diagram_html, height=600, scrolling=True)
        
        # 빠른 편집 옵션
        st.subheader("🔧 빠른 편집")
        colA, colB = st.columns(2)
        with colA:
            if st.session_state.nodes:
                del_id = st.selectbox("삭제할 노드", ["—"] + [n["id"] for n in st.session_state.nodes], index=0)
                if del_id != "—" and st.button("노드 삭제"):
                    st.session_state.nodes = [n for n in st.session_state.nodes if n["id"] != del_id]
                    st.session_state.edges = [e for e in st.session_state.edges if e[0]!=del_id and e[1]!=del_id]
                    if del_id in st.session_state.inputs: st.session_state.inputs.remove(del_id)
                    if del_id in st.session_state.outputs: st.session_state.outputs.remove(del_id)
                    if del_id in st.session_state.tensor_shapes: del st.session_state.tensor_shapes[del_id]
                    st.rerun()
        
        with colB:
            if st.session_state.edges:
                idx = st.number_input("삭제할 엣지 index", min_value=0, max_value=max(0,len(st.session_state.edges)-1), value=0, step=1)
                if st.button("엣지 삭제"):
                    st.session_state.edges.pop(idx)
                    st.rerun()
    else:
        st.info("노드를 추가하여 네트워크를 구성하세요.")
        st.image("https://via.placeholder.com/800x400/667eea/ffffff?text=TorchCanvas+Network+Designer", use_container_width=True)

with tab2:
    # 코드 생성 및 다운로드
    st.subheader("⚙️ PyTorch 코드 생성")
    
    if st.session_state.nodes and st.session_state.inputs and st.session_state.outputs:
        graph = {
            "version": "0.2",
            "metadata": {"name": "torchcanvas_ui"},
            "nodes": st.session_state.nodes,
            "edges": st.session_state.edges,
            "inputs": st.session_state.inputs,
            "outputs": st.session_state.outputs,
        }
        
        try:
            code_str = export_graph_to_python(graph, class_name="ExportedModel")
            
            # 코드 미리보기
            st.code(code_str, language="python")
            
            # 다운로드 및 테스트
            col1, col2 = st.columns([1,1])
            with col1:
                st.download_button(
                    "📥 exported_model.py 다운로드", 
                    data=code_str.encode("utf-8"),
                    file_name="exported_model.py", 
                    mime="text/x-python"
                )
            
            with col2:
                st.caption("🧪 스모크 테스트")
                b = st.number_input("Batch", min_value=1, value=4)
                C = st.number_input("Channels(C)", min_value=1, value=3)
                H = st.number_input("Height(H)", min_value=1, value=224)
                W = st.number_input("Width(W)", min_value=1, value=224)
                run = st.button("Forward 실행")
                if run:
                    ns: Dict[str, Any] = {}
                    try:
                        exec(code_str, ns, ns)
                        ExportedModel = ns["ExportedModel"]
                        import torch
                        m = ExportedModel()
                        x = torch.randn(int(b), int(C), int(H), int(W))
                        y = m({"inp": x})
                        st.success(f"✅ 성공! 출력 형태: {tuple(y.shape)}")
                    except Exception as e:
                        st.error(f"❌ 실행 에러: {e}")
        except Exception as e:
            st.error(f"코드 생성 에러: {e}")
    else:
        st.info("노드/엣지/입출력을 먼저 구성하세요.")

with tab3:
    # 상세 정보 (JSON, 노드/엣지 상세)
    st.subheader("📊 상세 정보")
    
    # 서브탭
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📋 노드 목록", "🔗 엣지 목록", "📄 Graph JSON"])
    
    with sub_tab1:
        if st.session_state.nodes:
            for i, node in enumerate(st.session_state.nodes):
                with st.expander(f"{i+1}. {node['id']} ({node['type']})"):
                    st.json(node)
        else:
            st.info("아직 노드가 없습니다.")
    
    with sub_tab2:
        if st.session_state.edges:
            for i, edge in enumerate(st.session_state.edges):
                st.write(f"{i+1}. {edge[0]} → {edge[1]}")
        else:
            st.info("아직 엣지가 없습니다.")
    
    with sub_tab3:
        graph = {
            "version": "0.2",
            "metadata": {"name": "torchcanvas_ui"},
            "nodes": st.session_state.nodes,
            "edges": st.session_state.edges,
            "inputs": st.session_state.inputs,
            "outputs": st.session_state.outputs,
        }
        st.code(json.dumps(graph, indent=2, ensure_ascii=False), language="json")

with tab4:
    # 템플릿
    st.subheader("📋 템플릿 예시")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🏗️ VGG-16 템플릿 로드"):
            vgg16_template = {
                "version": "0.2",
                "metadata": {"name": "vgg16_template"},
                "nodes": [
                    {"id": "inp", "type": "Input", "params": {}},
                    {"id": "b1", "type": "VGGBlock", "params": {"c1": 64, "c2": 64, "kernel_size": 3, "use_lrn": False, "pool": True}},
                    {"id": "b2", "type": "VGGBlock", "params": {"c1": 128, "c2": 128, "kernel_size": 3, "use_lrn": False, "pool": True}},
                    {"id": "b3", "type": "VGGBlock", "params": {"c1": 256, "c2": 256, "kernel_size": 3, "use_lrn": False, "pool": True}},
                    {"id": "b4", "type": "VGGBlock", "params": {"c1": 512, "c2": 512, "kernel_size": 3, "use_lrn": False, "pool": True}},
                    {"id": "b5", "type": "VGGBlock", "params": {"c1": 512, "c2": 512, "kernel_size": 3, "use_lrn": False, "pool": True}},
                    {"id": "flat", "type": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
                    {"id": "fc1", "type": "Linear", "params": {"out_features": 4096, "bias": True}},
                    {"id": "relu1", "type": "ReLU", "params": {}},
                    {"id": "drop1", "type": "Dropout", "params": {"p": 0.5}},
                    {"id": "fc2", "type": "Linear", "params": {"out_features": 4096, "bias": True}},
                    {"id": "relu2", "type": "ReLU", "params": {}},
                    {"id": "drop2", "type": "Dropout", "params": {"p": 0.5}},
                    {"id": "fc3", "type": "Linear", "params": {"out_features": 1000, "bias": True}},
                ],
                "edges": [
                    ["inp", "b1"], ["b1", "b2"], ["b2", "b3"], ["b3", "b4"], ["b4", "b5"],
                    ["b5", "flat"], ["flat", "fc1"], ["fc1", "relu1"], ["relu1", "drop1"],
                    ["drop1", "fc2"], ["fc2", "relu2"], ["relu2", "drop2"], ["drop2", "fc3"]
                ],
                "inputs": ["inp"],
                "outputs": ["fc3"]
            }
            st.session_state.nodes = vgg16_template["nodes"]
            st.session_state.edges = vgg16_template["edges"]
            st.session_state.inputs = vgg16_template["inputs"]
            st.session_state.outputs = vgg16_template["outputs"]
            st.session_state.tensor_shapes.clear()
            st.success("VGG-16 템플릿이 로드되었습니다!")
            st.rerun()
    
    with col2:
        if st.button("🏗️ ResNet-18 템플릿 로드"):
            resnet18_template = {
                "version": "0.2",
                "metadata": {"name": "resnet18_template"},
                "nodes": [
                    {"id": "inp", "type": "Input", "params": {}},
                    {"id": "conv1", "type": "Conv2d", "params": {"out_channels": 64, "kernel_size": 7, "stride": 2, "padding": "same"}},
                    {"id": "bn1", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "relu1", "type": "ReLU", "params": {}},
                    {"id": "pool1", "type": "MaxPool2d", "params": {"kernel_size": 3, "stride": 2}},
                    {"id": "res1", "type": "ResidualBlock", "params": {"out_channels": 64, "kernel_size": 3, "stride": 1}},
                    {"id": "res2", "type": "ResidualBlock", "params": {"out_channels": 128, "kernel_size": 3, "stride": 2}},
                    {"id": "res3", "type": "ResidualBlock", "params": {"out_channels": 256, "kernel_size": 3, "stride": 2}},
                    {"id": "res4", "type": "ResidualBlock", "params": {"out_channels": 512, "kernel_size": 3, "stride": 2}},
                    {"id": "gap", "type": "MaxPool2d", "params": {"kernel_size": 7, "stride": 1}},
                    {"id": "flat", "type": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
                    {"id": "fc", "type": "Linear", "params": {"out_features": 1000, "bias": True}},
                ],
                "edges": [
                    ["inp", "conv1"], ["conv1", "bn1"], ["bn1", "relu1"], ["relu1", "pool1"],
                    ["pool1", "res1"], ["res1", "res2"], ["res2", "res3"], ["res3", "res4"],
                    ["res4", "gap"], ["gap", "flat"], ["flat", "fc"]
                ],
                "inputs": ["inp"],
                "outputs": ["fc"]
            }
            st.session_state.nodes = resnet18_template["nodes"]
            st.session_state.edges = resnet18_template["edges"]
            st.session_state.inputs = resnet18_template["inputs"]
            st.session_state.outputs = resnet18_template["outputs"]
            st.session_state.tensor_shapes.clear()
            st.success("ResNet-18 템플릿이 로드되었습니다!")
            st.rerun()
    
    st.divider()
    st.subheader("📚 템플릿 설명")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **VGG-16**
        - 13개 컨볼루션 레이어 + 3개 완전연결 레이어
        - 이미지 분류에 특화
        - 깊은 네트워크 구조
        """)
    
    with col2:
        st.markdown("""
        **ResNet-18**
        - 잔차 연결을 사용한 18층 네트워크
        - 그래디언트 소실 문제 해결
        - 효율적인 학습 가능
        """)
