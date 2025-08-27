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

# ---------- 유틸리티 함수들 ----------
def find_path_to_node(start_id: str, end_id: str, edges: List[List[str]]) -> List[str]:
    """두 노드 간의 경로를 찾는 함수 (BFS 사용)"""
    if start_id == end_id:
        return [start_id]
    
    # 그래프를 인접 리스트로 변환
    graph = {}
    for src, dst in edges:
        if src not in graph:
            graph[src] = []
        graph[src].append(dst)
    
    # BFS로 경로 찾기
    queue = [(start_id, [start_id])]
    visited = set()
    
    while queue:
        current, path = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        if current == end_id:
            return path
        
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    
    return []  # 경로를 찾지 못한 경우

# ---------- 텐서 형태 추정 함수 ----------
def estimate_tensor_shape(node_id: str, node_type: str, params: dict, input_shapes: List[tuple]) -> tuple:
    """노드의 출력 텐서 형태를 추정"""
    if not input_shapes:
        return (1, 3, 224, 224)  # 기본 입력 형태
    
    input_shape = input_shapes[0]  # 첫 번째 입력 사용
    
    if node_type == "Input":
        return input_shape
    
    elif node_type == "Conv1d":
        B, C, T = input_shape
        out_channels = params.get("out_channels", 64)
        kernel_size = params.get("kernel_size", 3)
        stride = params.get("stride", 1)
        padding = params.get("padding", "same")
        
        if padding == "same":
            pad = kernel_size // 2
        else:
            pad = int(padding)
        
        T_out = (T + 2 * pad - kernel_size) // stride + 1
        return (B, out_channels, T_out)
    
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
    
    elif node_type == "MaxPool1d":
        B, C, T = input_shape
        kernel_size = params.get("kernel_size", 2)
        stride = params.get("stride", kernel_size)
        
        T_out = (T - kernel_size) // stride + 1
        return (B, C, T_out)
    
    elif node_type == "MaxPool2d":
        B, C, H, W = input_shape
        kernel_size = params.get("kernel_size", 2)
        stride = params.get("stride", kernel_size)
        
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        return (B, C, H_out, W_out)
    
    elif node_type == "AvgPool2d":
        B, C, H, W = input_shape
        kernel_size = params.get("kernel_size", 2)
        stride = params.get("stride", kernel_size)
        
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        return (B, C, H_out, W_out)
    
    elif node_type == "ReLU":
        return input_shape  # 형태 유지
    
    elif node_type == "Sigmoid":
        return input_shape  # 형태 유지
    
    elif node_type == "Tanh":
        return input_shape  # 형태 유지
    
    elif node_type == "Dropout":
        return input_shape  # 형태 유지
    
    elif node_type == "BatchNorm1d":
        return input_shape  # 형태 유지
    
    elif node_type == "BatchNorm2d":
        return input_shape  # 형태 유지
    
    elif node_type == "Flatten":
        B = input_shape[0]
        start_dim = params.get("start_dim", 1)
        end_dim = params.get("end_dim", -1)
        
        if end_dim == -1:
            end_dim = len(input_shape) - 1
        
        # Flatten 계산
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            if i < len(input_shape):
                flattened_size *= input_shape[i]
        
        result_shape = list(input_shape[:start_dim]) + [flattened_size]
        return tuple(result_shape)
    
    elif node_type == "Linear":
        B = input_shape[0]
        out_features = params.get("out_features", 1000)
        return (B, out_features)
    
    elif node_type == "GRUBlock":
        B, T, H = input_shape
        hidden_size = params.get("hidden_size", 128)
        bidirectional = params.get("bidirectional", True)
        out_mode = params.get("out", "last")
        
        if bidirectional:
            hidden_size *= 2
        
        if out_mode == "seq":
            return (B, T, hidden_size)
        else:
            return (B, hidden_size)
    
    elif node_type == "SEBlock":
        return input_shape  # 형태 유지
    
    elif node_type == "ResidualBlock":
        B, C, H, W = input_shape
        out_channels = params.get("out_channels", C)
        return (B, out_channels, H, W)
    
    elif node_type == "VGGBlock":
        B, C, H, W = input_shape
        c2 = params.get("c2", C)
        pool = params.get("pool", True)
        
        if pool:
            H = H // 2
            W = W // 2
        
        return (B, c2, H, W)
    
    # 기본적으로 입력 형태 유지
    return input_shape

# ---------- 모델 통계 계산 함수 ----------
def calculate_model_statistics(nodes: List[dict], edges: List[List[str]], tensor_shapes: Dict[str, tuple]) -> dict:
    """모델의 통계 정보 계산"""
    stats = {
        "total_layers": len(nodes),
        "total_connections": len(edges),
        "estimated_params": 0,
        "estimated_memory_mb": 0,
        "input_shapes": [],
        "output_shapes": [],
        "layer_types": {},
        "compatibility_issues": []
    }
    
    # 레이어 타입별 카운트
    for node in nodes:
        node_type = node["type"]
        stats["layer_types"][node_type] = stats["layer_types"].get(node_type, 0) + 1
    
    # 파라미터 수 추정
    for node in nodes:
        node_type = node["type"]
        params = node.get("params", {})
        
        if node_type == "Conv1d":
            out_channels = params.get("out_channels", 64)
            kernel_size = params.get("kernel_size", 3)
            # 입력 채널은 이전 레이어에서 추정
            in_channels = 3  # 기본값
            stats["estimated_params"] += in_channels * out_channels * kernel_size + out_channels
        
        elif node_type == "Conv2d":
            out_channels = params.get("out_channels", 64)
            kernel_size = params.get("kernel_size", 3)
            in_channels = 3  # 기본값
            stats["estimated_params"] += in_channels * out_channels * kernel_size * kernel_size + out_channels
        
        elif node_type == "Linear":
            out_features = params.get("out_features", 1000)
            in_features = 512  # 기본값
            stats["estimated_params"] += in_features * out_features + out_features
        
        elif node_type == "BatchNorm1d":
            stats["estimated_params"] += 64 * 2  # weight + bias
        
        elif node_type == "BatchNorm2d":
            stats["estimated_params"] += 64 * 2  # weight + bias
    
    # 메모리 사용량 추정 (MB)
    stats["estimated_memory_mb"] = stats["estimated_params"] * 4 / (1024 * 1024)  # float32 기준
    
    # 호환성 검사
    for edge in edges:
        src, dst = edge
        src_node = next((n for n in nodes if n["id"] == src), None)
        dst_node = next((n for n in nodes if n["id"] == dst), None)
        
        if src_node and dst_node:
            src_type = src_node["type"]
            dst_type = dst_node["type"]
            
            # 텐서 형태 호환성 검사
            if src in tensor_shapes and dst in tensor_shapes:
                src_shape = tensor_shapes[src]
                dst_shape = tensor_shapes[dst]
                
                # 간단한 호환성 검사
                if len(src_shape) != len(dst_shape):
                    stats["compatibility_issues"].append(
                        f"형태 불일치: {src}({src_shape}) → {dst}({dst_shape})"
                    )
    
    return stats

# ---------- 개선된 시각적 네트워크 다이어그램 생성 ----------
def create_network_diagram(nodes: List[dict], edges: List[List[str]], tensor_shapes: Dict[str, tuple]) -> str:
    """HTML/CSS로 네트워크 다이어그램 생성 (향상된 연결 시스템)"""
    
    # 연결 타입 분석
    def analyze_connections():
        connection_types = {
            "sequential": [],  # 순차 연결
            "skip": [],       # 스킵 커넥션 (ResNet 스타일)
            "branch": [],     # 분기 (여러 입력)
            "merge": [],      # 병합 (여러 출력)
            "complex": []     # 복잡한 연결
        }
        
        # 각 노드의 입출력 연결 분석
        node_connections = {}
        for node in nodes:
            node_id = node["id"]
            incoming = [src for src, dst in edges if dst == node_id]
            outgoing = [dst for src, dst in edges if src == node_id]
            node_connections[node_id] = {
                "incoming": incoming,
                "outgoing": outgoing,
                "fan_in": len(incoming),
                "fan_out": len(outgoing)
            }
        
        # 연결 타입 분류 (개선된 로직)
        for src, dst in edges:
            src_conn = node_connections[src]
            dst_conn = node_connections[dst]
            
            # 스킵 커넥션 감지 (ResNet 스타일)
            # Add 노드로 들어가는 연결 중 하나는 스킵 커넥션일 가능성이 높음
            dst_node = next((n for n in nodes if n["id"] == dst), None)
            if dst_node and dst_node["type"] == "Add":
                # Add 노드로 들어가는 연결은 스킵 커넥션으로 분류
                connection_types["skip"].append((src, dst))
            elif src_conn["fan_out"] == 1 and dst_conn["fan_in"] == 1:
                # 간단한 순차 연결
                connection_types["sequential"].append((src, dst))
            elif dst_conn["fan_in"] > 1:
                # 여러 입력을 받는 노드 (병합)
                connection_types["merge"].append((src, dst))
            elif src_conn["fan_out"] > 1:
                # 여러 출력을 가진 노드 (분기)
                connection_types["branch"].append((src, dst))
            else:
                connection_types["complex"].append((src, dst))
        
        return connection_types, node_connections
    
    connection_types, node_connections = analyze_connections()
    
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
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        margin: 20px 0;
        position: relative;
        min-height: 500px;
    }
    .layer {
        background: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        position: relative;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border-left: 5px solid;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        cursor: pointer;
        user-select: none;
    }
    .layer:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }
    .layer.connection-source {
        border: 3px solid #28a745;
        box-shadow: 0 0 25px rgba(40, 167, 69, 0.6);
        transform: scale(1.05);
    }
    .layer.connection-target {
        border: 3px solid #007bff;
        box-shadow: 0 0 25px rgba(0, 123, 255, 0.6);
        transform: scale(1.05);
    }
    .layer-header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }
    .layer-icon {
        font-size: 28px;
        margin-right: 12px;
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
        font-family: 'Courier New', monospace;
    }
    .tensor-shape {
        font-family: 'Courier New', monospace;
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        padding: 10px 15px;
        border-radius: 10px;
        margin: 12px 0;
        font-size: 13px;
        font-weight: bold;
        border: 2px solid #dee2e6;
    }
    .connection-info {
        font-size: 11px;
        color: #6c757d;
        margin-top: 10px;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 3px solid #dee2e6;
    }
    .connection-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 10px;
        padding: 4px 8px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-transform: uppercase;
    }
    .connection-badge.sequential { background-color: #28a745; }
    .connection-badge.skip { background-color: #ffc107; }
    .connection-badge.branch { background-color: #17a2b8; }
    .connection-badge.merge { background-color: #e83e8c; }
    .connection-badge.complex { background-color: #6f42c1; }
    
    .category-badge {
        position: absolute;
        top: 10px;
        left: 10px;
        font-size: 10px;
        padding: 4px 8px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-transform: uppercase;
    }
    .params {
        font-size: 11px;
        color: #6c757d;
        margin-top: 10px;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 6px;
        border-left: 3px solid #dee2e6;
        max-height: 60px;
        overflow-y: auto;
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
        stroke-width: 3;
        fill: none;
        opacity: 0.8;
        marker-end: url(#arrowhead);
    }
    .connection-line.sequential {
        stroke: #28a745;
        stroke-dasharray: none;
    }
    .connection-line.skip {
        stroke: #ffc107;
        stroke-dasharray: 10,5;
    }
    .connection-line.branch {
        stroke: #17a2b8;
        stroke-dasharray: 5,5;
    }
    .connection-line.merge {
        stroke: #e83e8c;
        stroke-dasharray: 15,5;
    }
    .connection-line.complex {
        stroke: #6f42c1;
        stroke-dasharray: 20,10,5,10;
    }
    
    .layer {
        position: relative;
        z-index: 2;
    }
    
    .connection-instructions {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        border: 2px dashed #667eea;
    }
    .connection-instructions h4 {
        color: #667eea;
        margin: 0 0 15px 0;
        font-size: 18px;
    }
    .connection-instructions p {
        margin: 8px 0;
        font-size: 14px;
        color: #6c757d;
    }
    
    .connection-stats {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    }
    .connection-stats h4 {
        color: #667eea;
        margin: 0 0 15px 0;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    .stat-item {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stat-label {
        font-size: 12px;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: bold;
    }
    .stat-value {
        font-size: 18px;
        font-weight: bold;
        color: #667eea;
        margin-top: 5px;
    }
    
    .warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        color: #856404;
    }
    
    .residual-highlight {
        background: linear-gradient(45deg, #ffc107, #ff8c00);
        color: white;
        border: none;
        box-shadow: 0 0 20px rgba(255, 193, 7, 0.3);
    }
    
    .skip-connection {
        position: relative;
    }
    .skip-connection::before {
        content: '';
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        border: 2px dashed #ffc107;
        border-radius: 20px;
        opacity: 0.5;
        pointer-events: none;
    }
    
    .skip-source {
        position: relative;
    }
    .skip-source::after {
        content: '⏭️';
        position: absolute;
        top: -5px;
        right: -5px;
        background: #ffc107;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        box-shadow: 0 2px 5px rgba(255, 193, 7, 0.5);
    }
    </style>
    <div class="network-container">
    """
    
    # 연결 안내 메시지
    html += """
    <div class="connection-instructions">
        <h4>🔗 향상된 연결 시스템</h4>
        <p>• <strong>순차 연결</strong>: 기본적인 레이어 간 연결 (실선)</p>
        <p>• <strong>스킵 커넥션</strong>: ResNet 스타일의 잔차 연결 (점선, 곡선)</p>
        <p>• <strong>분기/병합</strong>: 여러 입력/출력을 가진 복잡한 구조 (다양한 점선)</p>
        <p>• <strong>시각적 표시</strong>: Add 노드는 주황색, 스킵 소스는 ⏭️ 아이콘</p>
        <p>• <strong>클릭 연결</strong>: 레이어를 클릭하여 연결 타입 선택 후 연결</p>
    </div>
    """
    
    # 연결 통계 표시
    total_edges = len(edges)
    sequential_count = len(connection_types["sequential"])
    skip_count = len(connection_types["skip"])
    branch_count = len(connection_types["branch"])
    merge_count = len(connection_types["merge"])
    complex_count = len(connection_types["complex"])
    
    html += f"""
    <div class="connection-stats">
        <h4>📊 연결 분석 (향상된 시스템)</h4>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">총 연결</div>
                <div class="stat-value">{total_edges}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">순차 연결</div>
                <div class="stat-value">{sequential_count}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">스킵 커넥션</div>
                <div class="stat-value">{skip_count}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">분기</div>
                <div class="stat-value">{branch_count}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">병합</div>
                <div class="stat-value">{merge_count}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">복잡한 연결</div>
                <div class="stat-value">{complex_count}</div>
            </div>
        </div>
    </div>
    """
    
    # 노드들을 그리드 형태로 배치
    html += '<div class="layers-grid" style="position: relative;">'
    
    # SVG 연결선을 위한 컨테이너
    html += '''
    <svg class="connection-lines" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <marker id="arrowhead" markerWidth="12" markerHeight="8" 
                    refX="11" refY="4" orient="auto">
                <polygon points="0 0, 12 4, 0 8" fill="#667eea" />
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
        for src, dst in edges:
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
        
        # 연결 정보 분석
        conn_info = node_connections[node_id]
        connection_info = f"입력: {conn_info['fan_in']}개, 출력: {conn_info['fan_out']}개"
        
        # 연결 타입 결정
        connection_type = "sequential"
        if conn_info["fan_in"] > 1:
            connection_type = "merge"
        elif conn_info["fan_out"] > 1:
            connection_type = "branch"
        
        # ResNet 스타일 노드 감지 (개선된 로직)
        is_residual = node_type == "ResidualBlock" or node_type == "Add"
        is_skip_connection = False
        
        # Add 노드이거나 여러 입력을 받는 노드인 경우 스킵 커넥션으로 간주
        if node_type == "Add" or conn_info["fan_in"] > 1:
            is_skip_connection = True
        
        # 스킵 커넥션의 시작점도 감지
        is_skip_source = False
        if conn_info["fan_out"] > 1:
            # 여러 출력을 가진 노드 중 Add 노드로 연결되는 경우
            for dst in conn_info["outgoing"]:
                dst_node = next((n for n in nodes if n["id"] == dst), None)
                if dst_node and dst_node["type"] == "Add":
                    is_skip_source = True
                    break
        
        # 노드 위치 계산 (개선된 그리드 레이아웃)
        grid_col = i % 3
        grid_row = i // 3
        x_pos = grid_col * 320 + 160
        y_pos = grid_row * 220 + 120
        
        node_positions[node_id] = (x_pos, y_pos)
        
        # 특별한 스타일 클래스 추가
        extra_classes = []
        if is_residual:
            extra_classes.append("residual-highlight")
        if is_skip_connection:
            extra_classes.append("skip-connection")
        if is_skip_source:
            extra_classes.append("skip-source")
        
        extra_class_str = " " + " ".join(extra_classes) if extra_classes else ""
        
        html += f"""
        <div class="layer{extra_class_str}" style="border-left-color: {category_color};" data-node-id="{node_id}" onclick="handleLayerClick('{node_id}')">
            <div class="category-badge" style="background-color: {category_color};">{category_name}</div>
            <div class="connection-badge {connection_type}">{connection_type}</div>
            <div class="layer-header">
                <div class="layer-icon">{icon}</div>
                <div class="layer-name">{node_type}</div>
            </div>
            <div class="layer-id">ID: {node_id}</div>
            <div class="tensor-shape">[{', '.join(map(str, output_shape))}]</div>
            <div class="connection-info">{connection_info}</div>
            {f'<div class="params">{param_str}</div>' if param_str else ''}
        </div>
        """
    
    # 연결선 그리기 (개선된 스타일링)
    for src, dst in edges:
        if src in node_positions and dst in node_positions:
            x1, y1 = node_positions[src]
            x2, y2 = node_positions[dst]
            
            # 연결 타입 결정
            line_type = "sequential"
            if (src, dst) in connection_types["skip"]:
                line_type = "skip"
            elif (src, dst) in connection_types["branch"]:
                line_type = "branch"
            elif (src, dst) in connection_types["merge"]:
                line_type = "merge"
            elif (src, dst) in connection_types["complex"]:
                line_type = "complex"
            
            # 곡선 연결선 (더 부드러운 베지어 곡선)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # 스킵 커넥션의 경우 더 곡선적인 경로
            if line_type == "skip":
                # 더 긴 곡선 경로 (베지어 곡선)
                ctrl1_x = x1 + (x2 - x1) * 0.4
                ctrl1_y = y1 - 50  # 위로 올라가는 곡선
                ctrl2_x = x2 - (x2 - x1) * 0.4
                ctrl2_y = y2 - 50
                path_d = f"M {x1} {y1} C {ctrl1_x} {ctrl1_y}, {ctrl2_x} {ctrl2_y}, {x2} {y2}"
            elif line_type == "branch":
                # 분기 연결은 더 짧은 곡선
                ctrl_x = x1 + (x2 - x1) * 0.2
                ctrl_y = y1 + (y2 - y1) * 0.2
                path_d = f"M {x1} {y1} Q {ctrl_x} {ctrl_y} {x2} {y2}"
            elif line_type == "merge":
                # 병합 연결은 더 긴 곡선
                ctrl_x = x1 + (x2 - x1) * 0.6
                ctrl_y = y1 + (y2 - y1) * 0.6
                path_d = f"M {x1} {y1} Q {ctrl_x} {ctrl_y} {x2} {y2}"
            else:
                # 일반적인 곡선
                path_d = f"M {x1} {y1} Q {mid_x} {y1} {mid_x} {mid_y} T {x2} {y2}"
            
            html += f'''
            <path class="connection-line {line_type}" 
                  d="{path_d}"
                  marker-end="url(#arrowhead)" />
            '''
    
    html += "</svg></div>"
    
    # JavaScript for enhanced drag and drop connections
    html += """
    <script>
    let connectionSource = null;
    let connectionTarget = null;
    let connectionMode = 'normal'; // 'normal', 'skip', 'branch', 'merge'
    
    function handleLayerClick(nodeId) {
        if (!connectionSource) {
            // Start connection
            connectionSource = nodeId;
            document.querySelector(`[data-node-id="${nodeId}"]`).classList.add('connection-source');
            console.log('Connection started from:', nodeId);
            
            // Show connection mode options
            showConnectionModeOptions();
        } else if (connectionSource === nodeId) {
            // Cancel connection if clicking same node
            cancelConnection();
        } else {
            // Complete connection
            connectionTarget = nodeId;
            document.querySelector(`[data-node-id="${nodeId}"]`).classList.add('connection-target');
            
            // Send connection to Streamlit with mode
            if (window.parent && window.parent.postMessage) {
                window.parent.postMessage({
                    type: 'add_connection',
                    source: connectionSource,
                    target: connectionTarget,
                    mode: connectionMode
                }, '*');
            }
            
            // Visual feedback
            setTimeout(() => {
                cancelConnection();
                // Trigger page reload to show new connection
                window.location.reload();
            }, 800);
        }
    }
    
    function showConnectionModeOptions() {
        // Create connection mode selector
        const selector = document.createElement('div');
        selector.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 1000;
            text-align: center;
        `;
        selector.innerHTML = `
            <h4>연결 타입 선택</h4>
            <button onclick="setConnectionMode('normal')" style="margin: 5px; padding: 8px 16px; border: none; border-radius: 5px; background: #28a745; color: white; cursor: pointer;">순차 연결</button>
            <button onclick="setConnectionMode('skip')" style="margin: 5px; padding: 8px 16px; border: none; border-radius: 5px; background: #ffc107; color: white; cursor: pointer;">스킵 커넥션</button>
            <button onclick="setConnectionMode('branch')" style="margin: 5px; padding: 8px 16px; border: none; border-radius: 5px; background: #17a2b8; color: white; cursor: pointer;">분기 연결</button>
            <button onclick="setConnectionMode('merge')" style="margin: 5px; padding: 8px 16px; border: none; border-radius: 5px; background: #e83e8c; color: white; cursor: pointer;">병합 연결</button>
            <button onclick="cancelConnection()" style="margin: 5px; padding: 8px 16px; border: none; border-radius: 5px; background: #6c757d; color: white; cursor: pointer;">취소</button>
        `;
        selector.id = 'connection-mode-selector';
        document.body.appendChild(selector);
    }
    
    function setConnectionMode(mode) {
        connectionMode = mode;
        document.getElementById('connection-mode-selector').remove();
        console.log('Connection mode set to:', mode);
    }
    
    function cancelConnection() {
        if (connectionSource) {
            document.querySelector(`[data-node-id="${connectionSource}"]`).classList.remove('connection-source');
        }
        if (connectionTarget) {
            document.querySelector(`[data-node-id="${connectionTarget}"]`).classList.remove('connection-target');
        }
        connectionSource = null;
        connectionTarget = null;
        connectionMode = 'normal';
        
        // Remove connection mode selector if exists
        const selector = document.getElementById('connection-mode-selector');
        if (selector) {
            selector.remove();
        }
    }
    
    // ESC key to cancel connection
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            cancelConnection();
        }
    });
    
    // Click outside to cancel connection
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.layer') && !event.target.closest('#connection-mode-selector') && connectionSource) {
            cancelConnection();
        }
    });
    
    // Add hover effects for better UX
    document.addEventListener('DOMContentLoaded', function() {
        const layers = document.querySelectorAll('.layer');
        layers.forEach(layer => {
            layer.addEventListener('mouseenter', function() {
                if (connectionSource && connectionSource !== this.dataset.nodeId) {
                    this.style.transform = 'scale(1.1)';
                    this.style.boxShadow = '0 0 30px rgba(0, 123, 255, 0.4)';
                    
                    // 연결 타입에 따른 색상 변경
                    if (connectionMode === 'skip') {
                        this.style.boxShadow = '0 0 30px rgba(255, 193, 7, 0.4)';
                    } else if (connectionMode === 'branch') {
                        this.style.boxShadow = '0 0 30px rgba(23, 162, 184, 0.4)';
                    } else if (connectionMode === 'merge') {
                        this.style.boxShadow = '0 0 30px rgba(232, 62, 140, 0.4)';
                    }
                }
            });
            
            layer.addEventListener('mouseleave', function() {
                if (connectionSource && connectionSource !== this.dataset.nodeId) {
                    this.style.transform = '';
                    this.style.boxShadow = '';
                }
            });
        });
    });
    </script>
    """
    
    html += "</div>"
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
st.sidebar.title("🎨 TorchCanvas — 향상된 Palette")

# 카테고리별 팔레트
selected_category = st.sidebar.selectbox(
    "카테고리 선택",
    list(CATEGORIES.keys()),
    format_func=lambda x: CATEGORIES[x],
    index=0
)

# 선택된 카테고리의 블록들 표시
st.sidebar.subheader(f"📂 {CATEGORIES[selected_category]} (향상된 블록)")

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
        
        # 블록 추가 버튼 (향상된 버전)
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
with st.sidebar.expander("⚙️ 고급 설정 (향상된 기능)"):
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
st.sidebar.subheader("🔗 연결 관리 (향상된 시스템)")
if st.session_state.nodes:
    opts = [n["id"] for n in st.session_state.nodes]
    with st.sidebar.form("add_edge_form"):
        src = st.selectbox("src", opts)
        dst = st.selectbox("dst", opts)
        add_e = st.form_submit_button("엣지 추가")
        if add_e:
            st.session_state.edges.append([src, dst]); st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📥📤 입력/출력 설정 (향상된 시스템)")
all_ids = [n["id"] for n in st.session_state.nodes]
st.session_state.inputs = st.sidebar.multiselect("inputs", all_ids, default=st.session_state.inputs)
st.session_state.outputs = st.sidebar.multiselect("outputs", all_ids, default=st.session_state.outputs)

st.sidebar.divider()
if st.sidebar.button("🔄 그래프 초기화 (향상된 시스템)"):
    st.session_state.nodes.clear(); st.session_state.edges.clear()
    st.session_state.inputs.clear(); st.session_state.outputs.clear()
    st.session_state.tensor_shapes.clear()
    st.rerun()

# ---------- 메인: 탭 기반 인터페이스 ----------
st.title("TorchCanvas — 향상된 시각적 신경망 디자이너")

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["🎨 네트워크 시각화 (향상됨)", "⚙️ 코드 생성", "📊 상세 정보", "📋 템플릿"])

with tab1:
    # 네트워크 다이어그램 표시 (더 큰 크기)
    if st.session_state.nodes:
        st.subheader("🔍 네트워크 아키텍처 시각화 (향상된 연결 시스템)")
        
        # 모델 통계 계산
        model_stats = calculate_model_statistics(
            st.session_state.nodes, 
            st.session_state.edges, 
            st.session_state.tensor_shapes
        )
        
        # 네트워크 통계 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 레이어", model_stats["total_layers"])
        with col2:
            st.metric("연결 수", model_stats["total_connections"])
        with col3:
            st.metric("추정 파라미터", f"{model_stats['estimated_params']:,}")
        with col4:
            st.metric("메모리 사용량", f"{model_stats['estimated_memory_mb']:.1f}MB")
        
        # 레이어 타입별 통계
        if model_stats["layer_types"]:
            st.subheader("📊 레이어 타입별 통계")
            layer_cols = st.columns(min(4, len(model_stats["layer_types"])))
            for i, (layer_type, count) in enumerate(model_stats["layer_types"].items()):
                with layer_cols[i % 4]:
                    st.metric(layer_type, count)
        
        # 호환성 문제 표시
        if model_stats["compatibility_issues"]:
            st.subheader("⚠️ 호환성 문제")
            for issue in model_stats["compatibility_issues"]:
                st.error(issue)
        
        # 더 큰 다이어그램 (개선된 버전 사용)
        diagram_html = create_network_diagram(
            st.session_state.nodes, 
            st.session_state.edges, 
            st.session_state.tensor_shapes
        )
        st.components.v1.html(diagram_html, height=600, scrolling=True)
        
        # 빠른 편집 옵션
        st.subheader("🔧 빠른 편집 (향상된 인터페이스)")
        colA, colB, colC = st.columns(3)
        
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
                # 더 직관적인 엣지 삭제
                edge_options = [f"{e[0]} → {e[1]}" for e in st.session_state.edges]
                selected_edge = st.selectbox("삭제할 연결", ["—"] + edge_options, index=0)
                if selected_edge != "—" and st.button("연결 삭제"):
                    edge_idx = edge_options.index(selected_edge)
                    st.session_state.edges.pop(edge_idx)
                    st.rerun()
        
        with colC:
            if st.button("🧹 연결 정리"):
                # 의미없는 연결들 제거
                nodes_by_id = {n["id"]: n for n in st.session_state.nodes}
                cleaned_edges = []
                
                for src, dst in st.session_state.edges:
                    # 존재하지 않는 노드 간 연결 제거
                    if src in nodes_by_id and dst in nodes_by_id:
                        # Input → Output 직접 연결 제거
                        if nodes_by_id[src]["type"] == "Input" and nodes_by_id[dst]["type"] == "Output":
                            continue
                        # 같은 타입 간 직접 연결 제거 (일부 경우)
                        if (nodes_by_id[src]["type"] == nodes_by_id[dst]["type"] and 
                            nodes_by_id[src]["type"] in ["Conv1d", "Conv2d", "BatchNorm1d", "BatchNorm2d"]):
                            continue
                        cleaned_edges.append([src, dst])
                
                removed_count = len(st.session_state.edges) - len(cleaned_edges)
                st.session_state.edges = cleaned_edges
                st.success(f"{removed_count}개의 의미없는 연결이 제거되었습니다! (향상된 연결 시스템)")
                st.rerun()
        
        # 자동 연결 기능
        st.subheader("🔗 향상된 연결 도구 (ResNet 스타일 지원)")
        colC, colD, colE = st.columns(3)
        
        with colC:
            if len(st.session_state.nodes) >= 2:
                if st.button("🔄 순차 연결"):
                    # 모든 노드를 순서대로 연결
                    nodes = st.session_state.nodes
                    new_edges = []
                    for i in range(len(nodes) - 1):
                        new_edges.append([nodes[i]["id"], nodes[i + 1]["id"]])
                    
                    # 기존 엣지와 중복 제거
                    existing_edges = set(tuple(edge) for edge in st.session_state.edges)
                    for edge in new_edges:
                        if tuple(edge) not in existing_edges:
                            st.session_state.edges.append(edge)
                    
                    st.success(f"{len(new_edges)}개의 순차 연결이 추가되었습니다! (향상된 연결 시스템)")
                    st.rerun()
        
        with colD:
            if len(st.session_state.nodes) >= 2:
                if st.button("🎯 스마트 연결"):
                    # 타입별로 적절한 연결 생성
                    nodes = st.session_state.nodes
                    new_edges = []
                    
                    # Input → Conv → Norm → Activation → Pool → Linear → Output 패턴
                    type_order = ["Input", "Conv1d", "Conv2d", "BatchNorm1d", "BatchNorm2d", 
                                 "ReLU", "Sigmoid", "Tanh", "MaxPool1d", "MaxPool2d", 
                                 "AvgPool2d", "Flatten", "Linear", "Dropout", "Output"]
                    
                    # 타입별로 노드 그룹화
                    nodes_by_type = {}
                    for node in nodes:
                        node_type = node["type"]
                        if node_type not in nodes_by_type:
                            nodes_by_type[node_type] = []
                        nodes_by_type[node_type].append(node)
                    
                    # 순서대로 연결
                    prev_nodes = []
                    for target_type in type_order:
                        if target_type in nodes_by_type:
                            current_nodes = nodes_by_type[target_type]
                            
                            # 이전 노드들과 연결
                            for prev_node in prev_nodes:
                                for current_node in current_nodes:
                                    new_edges.append([prev_node["id"], current_node["id"]])
                            
                            prev_nodes = current_nodes
                    
                    # 기존 엣지와 중복 제거
                    existing_edges = set(tuple(edge) for edge in st.session_state.edges)
                    added_count = 0
                    for edge in new_edges:
                        if tuple(edge) not in existing_edges:
                            st.session_state.edges.append(edge)
                            added_count += 1
                    
                    st.success(f"{added_count}개의 스마트 연결이 추가되었습니다! (향상된 연결 시스템)")
                    st.rerun()
        
        with colE:
            if len(st.session_state.nodes) >= 3:
                if st.button("⏭️ ResNet 스킵 커넥션"):
                    # ResNet 스타일의 스킵 커넥션 생성
                    nodes = st.session_state.nodes
                    new_edges = []
                    
                    # Conv 레이어들을 찾아서 스킵 커넥션 생성
                    conv_nodes = [n for n in nodes if n["type"] in ["Conv1d", "Conv2d"]]
                    
                    for i in range(len(conv_nodes) - 1):
                        current_conv = conv_nodes[i]
                        next_conv = conv_nodes[i + 1]
                        
                        # 현재 Conv와 다음 Conv 사이에 Add 노드가 있는지 확인
                        add_nodes = [n for n in nodes if n["type"] == "Add" and 
                                   n["id"] not in [edge[1] for edge in st.session_state.edges]]
                        
                        if add_nodes:
                            add_node = add_nodes[0]
                            # 스킵 커넥션: current_conv → add_node, 그리고 다음 레이어 → add_node
                            new_edges.append([current_conv["id"], add_node["id"]])
                            
                            # 다음 Conv까지의 경로 찾기
                            path_to_next = find_path_to_node(current_conv["id"], next_conv["id"], st.session_state.edges)
                            if path_to_next:
                                last_node_in_path = path_to_next[-1]
                                new_edges.append([last_node_in_path, add_node["id"]])
                    
                    # 기존 엣지와 중복 제거
                    existing_edges = set(tuple(edge) for edge in st.session_state.edges)
                    added_count = 0
                    for edge in new_edges:
                        if tuple(edge) not in existing_edges:
                            st.session_state.edges.append(edge)
                            added_count += 1
                    
                    st.success(f"{added_count}개의 ResNet 스킵 커넥션이 추가되었습니다! (향상된 연결 시스템)")
                    st.rerun()
        
        # 고급 연결 도구
        st.subheader("🔧 고급 연결 도구 (복잡한 구조 지원)")
        colF, colG, colH = st.columns(3)
        
        with colF:
            if st.button("🔄 분기 연결"):
                # 선택된 노드에서 여러 출력으로 분기
                if st.session_state.nodes:
                    source_node = st.selectbox("분기 시작 노드", [n["id"] for n in st.session_state.nodes])
                    target_nodes = st.multiselect("분기 대상 노드들", [n["id"] for n in st.session_state.nodes if n["id"] != source_node])
                    
                    if st.button("분기 연결 생성"):
                        new_edges = []
                        for target in target_nodes:
                            new_edges.append([source_node, target])
                        
                        # 기존 엣지와 중복 제거
                        existing_edges = set(tuple(edge) for edge in st.session_state.edges)
                        added_count = 0
                        for edge in new_edges:
                            if tuple(edge) not in existing_edges:
                                st.session_state.edges.append(edge)
                                added_count += 1
                        
                        st.success(f"{added_count}개의 분기 연결이 추가되었습니다! (향상된 연결 시스템)")
                        st.rerun()
        
        with colG:
            if st.button("🔗 병합 연결"):
                # 여러 노드에서 하나의 노드로 병합
                if st.session_state.nodes:
                    source_nodes = st.multiselect("병합 시작 노드들", [n["id"] for n in st.session_state.nodes])
                    target_node = st.selectbox("병합 대상 노드", [n["id"] for n in st.session_state.nodes if n["id"] not in source_nodes])
                    
                    if st.button("병합 연결 생성"):
                        new_edges = []
                        for source in source_nodes:
                            new_edges.append([source, target_node])
                        
                        # 기존 엣지와 중복 제거
                        existing_edges = set(tuple(edge) for edge in st.session_state.edges)
                        added_count = 0
                        for edge in new_edges:
                            if tuple(edge) not in existing_edges:
                                st.session_state.edges.append(edge)
                                added_count += 1
                        
                        st.success(f"{added_count}개의 병합 연결이 추가되었습니다! (향상된 연결 시스템)")
                        st.rerun()
        
        with colH:
            if st.button("🎯 패턴 연결"):
                # 미리 정의된 패턴으로 연결
                pattern = st.selectbox("연결 패턴", [
                    "Conv-BN-ReLU",
                    "Conv-BN-ReLU-Pool", 
                    "Linear-ReLU-Dropout",
                    "ResNet Block",
                    "Dense Block"
                ])
                
                if st.button("패턴 연결 생성"):
                    nodes = st.session_state.nodes
                    new_edges = []
                    
                    if pattern == "Conv-BN-ReLU":
                        # Conv → BN → ReLU 패턴 찾기
                        conv_nodes = [n for n in nodes if n["type"] in ["Conv1d", "Conv2d"]]
                        bn_nodes = [n for n in nodes if n["type"] in ["BatchNorm1d", "BatchNorm2d"]]
                        relu_nodes = [n for n in nodes if n["type"] == "ReLU"]
                        
                        for conv in conv_nodes:
                            for bn in bn_nodes:
                                for relu in relu_nodes:
                                    new_edges.extend([[conv["id"], bn["id"]], [bn["id"], relu["id"]]])
                    
                    elif pattern == "ResNet Block":
                        # ResNet 블록 패턴: Conv-BN-ReLU-Conv-BN + Add
                        conv_nodes = [n for n in nodes if n["type"] in ["Conv1d", "Conv2d"]]
                        bn_nodes = [n for n in nodes if n["type"] in ["BatchNorm1d", "BatchNorm2d"]]
                        relu_nodes = [n for n in nodes if n["type"] == "ReLU"]
                        add_nodes = [n for n in nodes if n["type"] == "Add"]
                        
                        # 2개씩 그룹화하여 ResNet 블록 생성
                        for i in range(0, len(conv_nodes) - 1, 2):
                            if i + 1 < len(conv_nodes):
                                conv1, conv2 = conv_nodes[i], conv_nodes[i + 1]
                                bn1, bn2 = bn_nodes[i] if i < len(bn_nodes) else None, bn_nodes[i + 1] if i + 1 < len(bn_nodes) else None
                                relu = relu_nodes[i // 2] if i // 2 < len(relu_nodes) else None
                                add = add_nodes[i // 2] if i // 2 < len(add_nodes) else None
                                
                                if all([conv1, conv2, bn1, bn2, relu, add]):
                                    new_edges.extend([
                                        [conv1["id"], bn1["id"]], [bn1["id"], relu["id"]],
                                        [relu["id"], conv2["id"]], [conv2["id"], bn2["id"]],
                                        [conv1["id"], add["id"]], [bn2["id"], add["id"]]
                                    ])
                    
                    # 기존 엣지와 중복 제거
                    existing_edges = set(tuple(edge) for edge in st.session_state.edges)
                    added_count = 0
                    for edge in new_edges:
                        if tuple(edge) not in existing_edges:
                            st.session_state.edges.append(edge)
                            added_count += 1
                    
                    st.success(f"{added_count}개의 {pattern} 패턴 연결이 추가되었습니다! (향상된 연결 시스템)")
                    st.rerun()
    else:
        st.info("노드를 추가하여 네트워크를 구성하세요. (향상된 연결 시스템 지원)")
        st.image("https://via.placeholder.com/800x400/667eea/ffffff?text=TorchCanvas+Enhanced+Network+Designer", use_container_width=True)

with tab2:
    # 코드 생성 및 다운로드
    st.subheader("⚙️ PyTorch 코드 생성 (향상된 시스템)")
    
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
                st.caption("🧪 스모크 테스트 (향상된 시스템)")
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
                        st.success(f"✅ 성공! 출력 형태: {tuple(y.shape)} (향상된 연결 시스템)")
                    except Exception as e:
                        st.error(f"❌ 실행 에러: {e} (향상된 연결 시스템)")
        except Exception as e:
            st.error(f"코드 생성 에러: {e} (향상된 연결 시스템)")
    else:
        st.info("노드/엣지/입출력을 먼저 구성하세요. (향상된 연결 시스템 지원)")

with tab3:
    # 상세 정보 (JSON, 노드/엣지 상세)
    st.subheader("📊 상세 정보 (향상된 시스템)")
    
    # 서브탭
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📋 노드 목록 (향상됨)", "🔗 엣지 목록 (향상됨)", "📄 Graph JSON"])
    
    with sub_tab1:
        if st.session_state.nodes:
            for i, node in enumerate(st.session_state.nodes):
                with st.expander(f"{i+1}. {node['id']} ({node['type']})"):
                    st.json(node)
        else:
            st.info("아직 노드가 없습니다. (향상된 연결 시스템 지원)")
    
    with sub_tab2:
        if st.session_state.edges:
            for i, edge in enumerate(st.session_state.edges):
                st.write(f"{i+1}. {edge[0]} → {edge[1]}")
        else:
            st.info("아직 엣지가 없습니다. (향상된 연결 시스템 지원)")
    
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
    st.subheader("📋 템플릿 예시 (향상된 연결 시스템)")
    
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
            st.success("VGG-16 템플릿이 로드되었습니다! (향상된 연결 시스템)")
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
                    
                    # 첫 번째 Residual Block (64 channels)
                    {"id": "res1_conv1", "type": "Conv2d", "params": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": "same"}},
                    {"id": "res1_bn1", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "res1_relu1", "type": "ReLU", "params": {}},
                    {"id": "res1_conv2", "type": "Conv2d", "params": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": "same"}},
                    {"id": "res1_bn2", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "res1_add", "type": "Add", "params": {}},
                    {"id": "res1_relu2", "type": "ReLU", "params": {}},
                    
                    # 두 번째 Residual Block (128 channels with stride)
                    {"id": "res2_conv1", "type": "Conv2d", "params": {"out_channels": 128, "kernel_size": 3, "stride": 2, "padding": "same"}},
                    {"id": "res2_bn1", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "res2_relu1", "type": "ReLU", "params": {}},
                    {"id": "res2_conv2", "type": "Conv2d", "params": {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": "same"}},
                    {"id": "res2_bn2", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "res2_proj", "type": "Conv2d", "params": {"out_channels": 128, "kernel_size": 1, "stride": 2, "padding": 0}},
                    {"id": "res2_add", "type": "Add", "params": {}},
                    {"id": "res2_relu2", "type": "ReLU", "params": {}},
                    
                    # 세 번째 Residual Block (256 channels)
                    {"id": "res3_conv1", "type": "Conv2d", "params": {"out_channels": 256, "kernel_size": 3, "stride": 2, "padding": "same"}},
                    {"id": "res3_bn1", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "res3_relu1", "type": "ReLU", "params": {}},
                    {"id": "res3_conv2", "type": "Conv2d", "params": {"out_channels": 256, "kernel_size": 3, "stride": 1, "padding": "same"}},
                    {"id": "res3_bn2", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "res3_proj", "type": "Conv2d", "params": {"out_channels": 256, "kernel_size": 1, "stride": 2, "padding": 0}},
                    {"id": "res3_add", "type": "Add", "params": {}},
                    {"id": "res3_relu2", "type": "ReLU", "params": {}},
                    
                    # 네 번째 Residual Block (512 channels)
                    {"id": "res4_conv1", "type": "Conv2d", "params": {"out_channels": 512, "kernel_size": 3, "stride": 2, "padding": "same"}},
                    {"id": "res4_bn1", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "res4_relu1", "type": "ReLU", "params": {}},
                    {"id": "res4_conv2", "type": "Conv2d", "params": {"out_channels": 512, "kernel_size": 3, "stride": 1, "padding": "same"}},
                    {"id": "res4_bn2", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "res4_proj", "type": "Conv2d", "params": {"out_channels": 512, "kernel_size": 1, "stride": 2, "padding": 0}},
                    {"id": "res4_add", "type": "Add", "params": {}},
                    {"id": "res4_relu2", "type": "ReLU", "params": {}},
                    
                    {"id": "gap", "type": "MaxPool2d", "params": {"kernel_size": 7, "stride": 1}},
                    {"id": "flat", "type": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
                    {"id": "fc", "type": "Linear", "params": {"out_features": 1000, "bias": True}},
                ],
                "edges": [
                    # Initial layers
                    ["inp", "conv1"], ["conv1", "bn1"], ["bn1", "relu1"], ["relu1", "pool1"],
                    
                    # First residual block
                    ["pool1", "res1_conv1"], ["res1_conv1", "res1_bn1"], ["res1_bn1", "res1_relu1"],
                    ["res1_relu1", "res1_conv2"], ["res1_conv2", "res1_bn2"],
                    ["pool1", "res1_add"], ["res1_bn2", "res1_add"], ["res1_add", "res1_relu2"],
                    
                    # Second residual block (with projection)
                    ["res1_relu2", "res2_conv1"], ["res2_conv1", "res2_bn1"], ["res2_bn1", "res2_relu1"],
                    ["res2_relu1", "res2_conv2"], ["res2_conv2", "res2_bn2"],
                    ["res1_relu2", "res2_proj"], ["res2_proj", "res2_add"], ["res2_bn2", "res2_add"],
                    ["res2_add", "res2_relu2"],
                    
                    # Third residual block (with projection)
                    ["res2_relu2", "res3_conv1"], ["res3_conv1", "res3_bn1"], ["res3_bn1", "res3_relu1"],
                    ["res3_relu1", "res3_conv2"], ["res3_conv2", "res3_bn2"],
                    ["res2_relu2", "res3_proj"], ["res3_proj", "res3_add"], ["res3_bn2", "res3_add"],
                    ["res3_add", "res3_relu2"],
                    
                    # Fourth residual block (with projection)
                    ["res3_relu2", "res4_conv1"], ["res4_conv1", "res4_bn1"], ["res4_bn1", "res4_relu1"],
                    ["res4_relu1", "res4_conv2"], ["res4_conv2", "res4_bn2"],
                    ["res3_relu2", "res4_proj"], ["res4_proj", "res4_add"], ["res4_bn2", "res4_add"],
                    ["res4_add", "res4_relu2"],
                    
                    # Final layers
                    ["res4_relu2", "gap"], ["gap", "flat"], ["flat", "fc"]
                ],
                "inputs": ["inp"],
                "outputs": ["fc"]
            }
            st.session_state.nodes = resnet18_template["nodes"]
            st.session_state.edges = resnet18_template["edges"]
            st.session_state.inputs = resnet18_template["inputs"]
            st.session_state.outputs = resnet18_template["outputs"]
            st.session_state.tensor_shapes.clear()
            st.success("ResNet-18 템플릿이 로드되었습니다! (실제 스킵 커넥션 포함, 향상된 연결 시스템)")
            st.rerun()
    
    # 추가 템플릿들
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🏗️ LSTM 텍스트 분류 템플릿 로드"):
            lstm_template = {
                "version": "0.2",
                "metadata": {"name": "lstm_text_classifier"},
                "nodes": [
                    {"id": "inp", "type": "Input", "params": {}},
                    {"id": "permute1", "type": "Permute_BCT_to_BTH", "params": {}},
                    {"id": "lstm1", "type": "GRUBlock", "params": {"hidden_size": 128, "num_layers": 2, "bidirectional": True, "out": "last"}},
                    {"id": "dropout1", "type": "Dropout", "params": {"p": 0.5}},
                    {"id": "fc1", "type": "Linear", "params": {"out_features": 64, "bias": True}},
                    {"id": "relu1", "type": "ReLU", "params": {}},
                    {"id": "dropout2", "type": "Dropout", "params": {"p": 0.3}},
                    {"id": "fc2", "type": "Linear", "params": {"out_features": 10, "bias": True}},
                ],
                "edges": [
                    ["inp", "permute1"], ["permute1", "lstm1"], ["lstm1", "dropout1"],
                    ["dropout1", "fc1"], ["fc1", "relu1"], ["relu1", "dropout2"], ["dropout2", "fc2"]
                ],
                "inputs": ["inp"],
                "outputs": ["fc2"]
            }
            st.session_state.nodes = lstm_template["nodes"]
            st.session_state.edges = lstm_template["edges"]
            st.session_state.inputs = lstm_template["inputs"]
            st.session_state.outputs = lstm_template["outputs"]
            st.session_state.tensor_shapes.clear()
            st.success("LSTM 텍스트 분류 템플릿이 로드되었습니다! (향상된 연결 시스템)")
            st.rerun()
    
    with col4:
        if st.button("🏗️ SE-ResNet 템플릿 로드"):
            se_resnet_template = {
                "version": "0.2",
                "metadata": {"name": "se_resnet_template"},
                "nodes": [
                    {"id": "inp", "type": "Input", "params": {}},
                    {"id": "conv1", "type": "Conv2d", "params": {"out_channels": 64, "kernel_size": 7, "stride": 2, "padding": "same"}},
                    {"id": "bn1", "type": "BatchNorm2d", "params": {"num_features": 0}},
                    {"id": "relu1", "type": "ReLU", "params": {}},
                    {"id": "pool1", "type": "MaxPool2d", "params": {"kernel_size": 3, "stride": 2}},
                    {"id": "res1", "type": "ResidualBlock", "params": {"out_channels": 64, "kernel_size": 3, "stride": 1}},
                    {"id": "se1", "type": "SEBlock", "params": {"reduction": 16}},
                    {"id": "res2", "type": "ResidualBlock", "params": {"out_channels": 128, "kernel_size": 3, "stride": 2}},
                    {"id": "se2", "type": "SEBlock", "params": {"reduction": 16}},
                    {"id": "gap", "type": "MaxPool2d", "params": {"kernel_size": 7, "stride": 1}},
                    {"id": "flat", "type": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
                    {"id": "fc", "type": "Linear", "params": {"out_features": 1000, "bias": True}},
                ],
                "edges": [
                    ["inp", "conv1"], ["conv1", "bn1"], ["bn1", "relu1"], ["relu1", "pool1"],
                    ["pool1", "res1"], ["res1", "se1"], ["se1", "res2"], ["res2", "se2"],
                    ["se2", "gap"], ["gap", "flat"], ["flat", "fc"]
                ],
                "inputs": ["inp"],
                "outputs": ["fc"]
            }
            st.session_state.nodes = se_resnet_template["nodes"]
            st.session_state.edges = se_resnet_template["edges"]
            st.session_state.inputs = se_resnet_template["inputs"]
            st.session_state.outputs = se_resnet_template["outputs"]
            st.session_state.tensor_shapes.clear()
            st.success("SE-ResNet 템플릿이 로드되었습니다! (향상된 연결 시스템)")
            st.rerun()
    
    st.divider()
    st.subheader("📚 템플릿 설명 (향상된 연결 시스템)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **VGG-16**
        - 13개 컨볼루션 레이어 + 3개 완전연결 레이어
        - 이미지 분류에 특화
        - 깊은 네트워크 구조
        - 순차적 연결 패턴
        """)
    
    with col2:
        st.markdown("""
        **ResNet-18**
        - 잔차 연결을 사용한 18층 네트워크
        - 그래디언트 소실 문제 해결
        - 실제 스킵 커넥션 구조 포함
        - 시각적으로 명확한 연결 표시
        """)
    
    with col3:
        st.markdown("""
        **LSTM 텍스트 분류**
        - GRU 기반 순환 신경망
        - 텍스트 분류에 특화
        - 양방향 처리 지원
        - 순열 연산 포함
        """)
    
    with col4:
        st.markdown("""
        **SE-ResNet**
        - Squeeze-and-Excitation 블록 포함
        - 채널 어텐션 메커니즘
        - 향상된 특징 표현
        - 복잡한 분기 구조
        """)
