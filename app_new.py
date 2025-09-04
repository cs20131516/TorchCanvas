# TorchCanvas - 새로운 접근 방식
# 드래그 앤 드롭 기반 신경망 디자이너
import streamlit as st
import streamlit.components.v1 as components
import json
from typing import Dict, List, Any, Tuple
import uuid

st.set_page_config(
    page_title="TorchCanvas - 신경망 디자이너",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 초기화 ----------
if 'nodes' not in st.session_state:
    st.session_state.nodes = []
if 'edges' not in st.session_state:
    st.session_state.edges = []
if 'selected_node' not in st.session_state:
    st.session_state.selected_node = None

# ---------- 노드 타입 정의 ----------
NODE_TYPES = {
    "Input": {
        "icon": "📥",
        "color": "#4CAF50",
        "description": "입력 레이어",
        "params": {}
    },
    "Conv2d": {
        "icon": "🔲",
        "color": "#2196F3",
        "description": "2D 컨볼루션",
        "params": {
            "out_channels": {"type": "int", "default": 64, "min": 1, "max": 1024},
            "kernel_size": {"type": "int", "default": 3, "min": 1, "max": 7},
            "stride": {"type": "int", "default": 1, "min": 1, "max": 5},
            "padding": {"type": "select", "options": ["same", "valid", "0", "1", "2"], "default": "same"}
        }
    },
    "ReLU": {
        "icon": "⚡",
        "color": "#FF9800",
        "description": "ReLU 활성화",
        "params": {}
    },
    "MaxPool2d": {
        "icon": "🔽",
        "color": "#9C27B0",
        "description": "2D 최대 풀링",
        "params": {
            "kernel_size": {"type": "int", "default": 2, "min": 2, "max": 8},
            "stride": {"type": "int", "default": 2, "min": 1, "max": 8}
        }
    },
    "BatchNorm2d": {
        "icon": "📊",
        "color": "#607D8B",
        "description": "2D 배치 정규화",
        "params": {}
    },
    "Linear": {
        "icon": "🔗",
        "color": "#795548",
        "description": "완전연결층",
        "params": {
            "out_features": {"type": "int", "default": 128, "min": 1, "max": 2048}
        }
    },
    "Dropout": {
        "icon": "🎲",
        "color": "#E91E63",
        "description": "드롭아웃",
        "params": {
            "p": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}
        }
    },
    "Output": {
        "icon": "📤",
        "color": "#F44336",
        "description": "출력 레이어",
        "params": {
            "num_classes": {"type": "int", "default": 10, "min": 1, "max": 1000}
        }
    },
    "Add": {
        "icon": "➕",
        "color": "#FF5722",
        "description": "덧셈 연산",
        "params": {}
    }
}

# ---------- 템플릿 정의 ----------
TEMPLATES = {
    "Simple CNN": {
        "description": "간단한 CNN",
        "nodes": [
            {"id": "input", "type": "Input", "x": 100, "y": 200},
            {"id": "conv1", "type": "Conv2d", "x": 300, "y": 200, "params": {"out_channels": 32, "kernel_size": 3}},
            {"id": "relu1", "type": "ReLU", "x": 500, "y": 200},
            {"id": "pool1", "type": "MaxPool2d", "x": 700, "y": 200},
            {"id": "conv2", "type": "Conv2d", "x": 900, "y": 200, "params": {"out_channels": 64, "kernel_size": 3}},
            {"id": "relu2", "type": "ReLU", "x": 1100, "y": 200},
            {"id": "pool2", "type": "MaxPool2d", "x": 1300, "y": 200},
            {"id": "linear", "type": "Linear", "x": 1500, "y": 200, "params": {"out_features": 128}},
            {"id": "output", "type": "Output", "x": 1700, "y": 200, "params": {"num_classes": 10}}
        ],
        "edges": [
            ["input", "conv1"], ["conv1", "relu1"], ["relu1", "pool1"],
            ["pool1", "conv2"], ["conv2", "relu2"], ["relu2", "pool2"],
            ["pool2", "linear"], ["linear", "output"]
        ]
    },
    "ResNet Block": {
        "description": "ResNet 블록",
        "nodes": [
            {"id": "input", "type": "Input", "x": 100, "y": 200},
            {"id": "conv1", "type": "Conv2d", "x": 300, "y": 150, "params": {"out_channels": 64, "kernel_size": 3}},
            {"id": "bn1", "type": "BatchNorm2d", "x": 500, "y": 150},
            {"id": "relu1", "type": "ReLU", "x": 700, "y": 150},
            {"id": "conv2", "type": "Conv2d", "x": 900, "y": 150, "params": {"out_channels": 64, "kernel_size": 3}},
            {"id": "bn2", "type": "BatchNorm2d", "x": 1100, "y": 150},
            {"id": "add", "type": "Add", "x": 1300, "y": 200},
            {"id": "relu2", "type": "ReLU", "x": 1500, "y": 200},
            {"id": "output", "type": "Output", "x": 1700, "y": 200}
        ],
        "edges": [
            ["input", "conv1"], ["conv1", "bn1"], ["bn1", "relu1"],
            ["relu1", "conv2"], ["conv2", "bn2"], ["bn2", "add"],
            ["input", "add"], ["add", "relu2"], ["relu2", "output"]
        ]
    }
}

# ---------- PyTorch 코드 생성 ----------
def generate_pytorch_code(nodes: List[Dict], edges: List[List[str]]) -> str:
    """노드와 엣지로부터 PyTorch 코드 생성"""
    
    # 노드 ID로 정렬
    node_dict = {node["id"]: node for node in nodes}
    
    # __init__ 메서드 생성
    init_lines = ["import torch", "import torch.nn as nn", "", "class Network(nn.Module):", "    def __init__(self):", "        super().__init__()"]
    
    for node in nodes:
        node_type = node["type"]
        node_id = node["id"]
        params = node.get("params", {})
        
        if node_type == "Input":
            continue
        elif node_type == "Conv2d":
            out_channels = params.get("out_channels", 64)
            kernel_size = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            padding = params.get("padding", "same")
            
            if padding == "same":
                padding_val = kernel_size // 2
            else:
                padding_val = int(padding)
            
            init_lines.append(f"        self.{node_id} = nn.Conv2d(in_channels, {out_channels}, {kernel_size}, stride={stride}, padding={padding_val})")
        
        elif node_type == "ReLU":
            init_lines.append(f"        self.{node_id} = nn.ReLU()")
        
        elif node_type == "MaxPool2d":
            kernel_size = params.get("kernel_size", 2)
            stride = params.get("stride", 2)
            init_lines.append(f"        self.{node_id} = nn.MaxPool2d({kernel_size}, stride={stride})")
        
        elif node_type == "BatchNorm2d":
            init_lines.append(f"        self.{node_id} = nn.BatchNorm2d(num_features)")
        
        elif node_type == "Linear":
            out_features = params.get("out_features", 128)
            init_lines.append(f"        self.{node_id} = nn.Linear(in_features, {out_features})")
        
        elif node_type == "Dropout":
            p = params.get("p", 0.5)
            init_lines.append(f"        self.{node_id} = nn.Dropout({p})")
        
        elif node_type == "Output":
            num_classes = params.get("num_classes", 10)
            init_lines.append(f"        self.{node_id} = nn.Linear(in_features, {num_classes})")
        
        elif node_type == "Add":
            # Add는 별도의 레이어가 아니므로 init에서 제외
            pass
    
    # forward 메서드 생성
    forward_lines = ["    def forward(self, x):"]
    
    # 입력 노드 찾기
    input_nodes = [node for node in nodes if node["type"] == "Input"]
    if input_nodes:
        forward_lines.append(f"        # Input: {input_nodes[0]['id']}")
    
    # 노드 실행 순서 결정 (간단한 위상정렬)
    in_degree = {node["id"]: 0 for node in nodes}
    for edge in edges:
        if len(edge) == 2:
            in_degree[edge[1]] += 1
    
    queue = [node["id"] for node in nodes if in_degree[node["id"]] == 0]
    execution_order = []
    
    while queue:
        current = queue.pop(0)
        execution_order.append(current)
        
        for edge in edges:
            if len(edge) == 2 and edge[0] == current:
                target = edge[1]
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)
    
    # forward 메서드 본문 생성
    for node_id in execution_order:
        if node_id in node_dict:
            node = node_dict[node_id]
            node_type = node["type"]
            
            if node_type == "Input":
                forward_lines.append(f"        # {node_id} (input)")
            elif node_type == "Add":
                # Add 노드의 경우 입력들을 찾아서 덧셈 수행
                inputs = [edge[0] for edge in edges if edge[1] == node_id]
                if len(inputs) >= 2:
                    forward_lines.append(f"        {node_id} = {inputs[0]} + {inputs[1]}")
                else:
                    forward_lines.append(f"        # {node_id} (add operation)")
            else:
                forward_lines.append(f"        x = self.{node_id}(x)")
    
    forward_lines.append("        return x")
    
    return "\n".join(init_lines + [""] + forward_lines)

# ---------- 시각화 HTML 생성 ----------
def create_visualization_html(nodes: List[Dict], edges: List[List[str]]) -> str:
    """드래그 앤 드롭 가능한 시각화 HTML 생성"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }}
            
            .canvas {{
                width: 100%;
                height: 800px;
                background: white;
                border: 2px dashed #ddd;
                border-radius: 10px;
                position: relative;
                overflow: visible;
                cursor: grab;
            }}
            
            .canvas:active {{
                cursor: grabbing;
            }}
            
            .canvas-container {{
                position: relative;
                overflow: hidden;
                width: 100%;
                height: 800px;
            }}
            
            .node {{
                position: absolute;
                width: 120px;
                height: 80px;
                border-radius: 10px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                cursor: move;
                user-select: none;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: all 0.2s ease;
                border: 2px solid transparent;
                z-index: 1;
            }}
            
            .node:hover {{
                transform: scale(1.05);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }}
            
            .node.selected {{
                border-color: #007bff;
                box-shadow: 0 0 0 3px rgba(0,123,255,0.25);
            }}
            
            .node-icon {{
                font-size: 24px;
                margin-bottom: 4px;
            }}
            
            .node-label {{
                font-size: 12px;
                font-weight: bold;
                text-align: center;
                color: white;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }}
            
            .connection {{
                position: absolute;
                pointer-events: none;
                z-index: 1;
            }}
            
            .connection-line {{
                stroke: #666;
                stroke-width: 3;
                fill: none;
                marker-end: url(#arrowhead);
                vector-effect: non-scaling-stroke;
                stroke-linecap: round;
                shape-rendering: geometricPrecision;
            }}
            
            .connection-line:hover {{
                stroke: #007bff;
                stroke-width: 4;
            }}
            
            .palette {{
                position: fixed;
                top: 20px;
                left: 20px;
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 1000;
            }}
            
            .palette-item {{
                display: flex;
                align-items: center;
                padding: 8px 12px;
                margin: 4px 0;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s ease;
                border: 1px solid #eee;
                user-select: none;
            }}
            
            .palette-item:hover {{
                background: #e3f2fd;
                transform: translateX(4px);
                border-color: #2196f3;
            }}
            
            .palette-item:active {{
                background: #bbdefb;
                transform: translateX(2px);
            }}
            
            .palette-icon {{
                font-size: 20px;
                margin-right: 8px;
            }}
            
            .palette-label {{
                font-size: 14px;
                font-weight: 500;
            }}
            
            .controls {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 1000;
            }}
            
            .btn {{
                background: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                margin: 4px;
                font-size: 14px;
                transition: all 0.2s ease;
            }}
            
            .btn:hover {{
                background: #0056b3;
                transform: translateY(-1px);
            }}
            
            .btn-danger {{
                background: #dc3545;
            }}
            
            .btn-danger:hover {{
                background: #c82333;
            }}
            
            .zoom-controls {{
                position: fixed;
                top: 20px;
                right: 200px;
                background: white;
                border-radius: 10px;
                padding: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 1000;
                display: flex;
                gap: 5px;
            }}
            
            .zoom-btn {{
                background: #6c757d;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.2s ease;
            }}
            
            .zoom-btn:hover {{
                background: #5a6268;
                transform: scale(1.05);
            }}
            
            .zoom-level {{
                background: #f8f9fa;
                color: #495057;
                border: 1px solid #dee2e6;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                min-width: 60px;
                text-align: center;
            }}
            
            .popup-btn {{
                background: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                margin: 4px;
                font-size: 14px;
                transition: all 0.2s ease;
            }}
            
            .popup-btn:hover {{
                background: #218838;
                transform: translateY(-1px);
            }}
        </style>
    </head>
    <body>
        <!-- 팔레트 -->
        <div class="palette">
            <h4 style="margin: 0 0 10px 0; color: #333;">노드 팔레트</h4>
            <p style="margin: 0 0 10px 0; font-size: 12px; color: #666;">클릭하거나 드래그하여 추가</p>
    """
    
    # 노드 타입별 팔레트 아이템 생성
    for node_type, config in NODE_TYPES.items():
        html += f"""
            <div class="palette-item" draggable="true" data-type="{node_type}">
                <div class="palette-icon">{config['icon']}</div>
                <div class="palette-label">{node_type}</div>
            </div>
        """
    
    html += """
        </div>
        
        <!-- 줌 컨트롤 -->
        <div class="zoom-controls">
            <button class="zoom-btn" onclick="zoomOut()">-</button>
            <div class="zoom-level" id="zoomLevel">100%</div>
            <button class="zoom-btn" onclick="zoomIn()">+</button>
            <button class="zoom-btn" onclick="resetZoom()">⌂</button>
        </div>
        
        <!-- 컨트롤 -->
        <div class="controls">
            <h4 style="margin: 0 0 10px 0; color: #333;">컨트롤</h4>
            <button class="btn" onclick="clearCanvas()">전체 삭제</button>
            <button class="btn" onclick="deleteSelected()">선택 삭제</button>
            <button class="btn" onclick="generateCode()">코드 생성</button>
            <button class="popup-btn" onclick="openPopup()" disabled>🪟 팝업 창 (개발중)</button>
        </div>
        
        <!-- 캔버스 컨테이너 -->
        <div class="canvas-container" id="canvasContainer">
            <div class="canvas" id="canvas">
                <svg style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 999;">
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
                        </marker>
                    </defs>
                </svg>
            </div>
        </div>
        
        <script>
            let nodes = """ + json.dumps(nodes) + """;
            let edges = """ + json.dumps(edges) + """;
            let selectedNode = null;
            let isDragging = false;
            let dragOffset = {x: 0, y: 0};
            
            // 줌/팬 변수
            let zoomLevel = 1;
            let panX = 0;
            let panY = 0;
            let isPanning = false;
            let panStart = {x: 0, y: 0};
            
            // 노드 렌더링
            function renderNodes() {
                console.log('Rendering nodes:', nodes);
                const canvas = document.getElementById('canvas');
                const existingNodes = canvas.querySelectorAll('.node');
                existingNodes.forEach(node => node.remove());
                
                nodes.forEach(node => {
                    console.log('Creating node:', node);
                    const nodeElement = document.createElement('div');
                    nodeElement.className = 'node';
                    nodeElement.style.left = node.x + 'px';
                    nodeElement.style.top = node.y + 'px';
                    nodeElement.style.backgroundColor = getNodeColor(node.type);
                    nodeElement.dataset.nodeId = node.id;
                    
                    nodeElement.innerHTML = '<div class="node-icon">' + getNodeIcon(node.type) + '</div><div class="node-label">' + node.id + '</div>';
                    
                    // 이벤트 리스너
                    nodeElement.addEventListener('mousedown', startDrag);
                    nodeElement.addEventListener('click', selectNode);
                    
                    canvas.appendChild(nodeElement);
                });
                
                renderConnections();
            }
            
            // 노드 테두리에서 연결점 계산
            function edgePointOnRect(cx, cy, w, h, tx, ty) {
                const dx = tx - cx, dy = ty - cy;
                const absDx = Math.abs(dx), absDy = Math.abs(dy);
                const halfW = w / 2, halfH = h / 2;
                
                // 사각형-중심에서 타겟방향으로 나가는 교차점
                if (absDx / halfW > absDy / halfH) {
                    // 좌우 변과 교차
                    const sx = dx > 0 ? cx + halfW : cx - halfW;
                    const t = (sx - cx) / dx;
                    return { x: sx, y: cy + dy * t };
                } else {
                    // 상하 변과 교차
                    const sy = dy > 0 ? cy + halfH : cy - halfH;
                    const t = (sy - cy) / dy;
                    return { x: cx + dx * t, y: sy };
                }
            }
            
            // 연결선 렌더링
            function renderConnections() {
                const svg = document.querySelector('svg');
                const existingLines = svg.querySelectorAll('.connection-line');
                existingLines.forEach(line => line.remove());
                
                edges.forEach(edge => {
                    if (edge.length === 2) {
                        const sourceNode = nodes.find(n => n.id === edge[0]);
                        const targetNode = nodes.find(n => n.id === edge[1]);
                        
                        if (sourceNode && targetNode) {
                            // 노드 중심 좌표
                            const sCenter = { x: sourceNode.x + 60, y: sourceNode.y + 40 };
                            const tCenter = { x: targetNode.x + 60, y: targetNode.y + 40 };
                            
                            // 노드 테두리에서의 연결점 계산
                            const sEdge = edgePointOnRect(sCenter.x, sCenter.y, 120, 80, tCenter.x, tCenter.y);
                            const tEdge = edgePointOnRect(tCenter.x, tCenter.y, 120, 80, sCenter.x, sCenter.y);
                            
                            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                            line.setAttribute('class', 'connection-line');
                            line.setAttribute('x1', sEdge.x);
                            line.setAttribute('y1', sEdge.y);
                            line.setAttribute('x2', tEdge.x);
                            line.setAttribute('y2', tEdge.y);
                            
                            svg.appendChild(line);
                        }
                    }
                });
            }
            
            // 노드 색상 가져오기
            function getNodeColor(type) {
                const colors = {
    """
    
    for node_type, config in NODE_TYPES.items():
        html += f'                    "{node_type}": "{config["color"]}",\n'
    
    html += """
                };
                return colors[type] || '#666';
            }
            
            // 노드 아이콘 가져오기
            function getNodeIcon(type) {
                const icons = {
    """
    
    for node_type, config in NODE_TYPES.items():
        html += f'                    "{node_type}": "{config["icon"]}",\n'
    
    html += """
                };
                return icons[type] || '❓';
            }
            
            // 드래그 시작
            function startDrag(e) {
                e.preventDefault();
                isDragging = true;
                selectedNode = e.target.closest('.node');
                
                const rect = selectedNode.getBoundingClientRect();
                const canvasRect = document.getElementById('canvas').getBoundingClientRect();
                
                dragOffset.x = e.clientX - rect.left;
                dragOffset.y = e.clientY - rect.top;
                
                document.addEventListener('mousemove', drag);
                document.addEventListener('mouseup', endDrag);
            }
            
            // 드래그 중
            function drag(e) {
                if (!isDragging || !selectedNode) return;
                
                const canvas = document.getElementById('canvas');
                const canvasRect = canvas.getBoundingClientRect();
                
                const x = e.clientX - canvasRect.left - dragOffset.x;
                const y = e.clientY - canvasRect.top - dragOffset.y;
                
                // 캔버스 경계 내로 제한
                const maxX = canvasRect.width - 120;
                const maxY = canvasRect.height - 80;
                
                const clampedX = Math.max(0, Math.min(x, maxX));
                const clampedY = Math.max(0, Math.min(y, maxY));
                
                selectedNode.style.left = clampedX + 'px';
                selectedNode.style.top = clampedY + 'px';
                
                // 노드 데이터 업데이트
                const nodeId = selectedNode.dataset.nodeId;
                const node = nodes.find(n => n.id === nodeId);
                if (node) {
                    node.x = clampedX;
                    node.y = clampedY;
                }
                
                renderConnections();
            }
            
            // 드래그 종료
            function endDrag() {
                isDragging = false;
                selectedNode = null;
                document.removeEventListener('mousemove', drag);
                document.removeEventListener('mouseup', endDrag);
            }
            
            // 노드 선택
            function selectNode(e) {
                e.stopPropagation();
                
                // 기존 선택 해제
                document.querySelectorAll('.node').forEach(node => {
                    node.classList.remove('selected');
                });
                
                // 새 노드 선택
                const node = e.target.closest('.node');
                node.classList.add('selected');
                selectedNode = node;
            }
            
            // 팔레트에서 드래그 시작 및 클릭 이벤트
            document.querySelectorAll('.palette-item').forEach(item => {
                item.addEventListener('dragstart', function(e) {
                    e.dataTransfer.setData('text/plain', this.dataset.type);
                    console.log('Drag started:', this.dataset.type);
                });
                
                // 클릭으로도 노드 추가 가능
                item.addEventListener('click', function() {
                    const nodeType = this.dataset.type;
                    console.log('Click to add node:', nodeType);
                    
                    // 캔버스 중앙에 노드 추가
                    const canvas = document.getElementById('canvas');
                    const rect = canvas.getBoundingClientRect();
                    const x = rect.width / 2 - 60;
                    const y = rect.height / 2 - 40;
                    
                    const nodeId = nodeType.toLowerCase() + '_' + Date.now();
                    const newNode = {
                        id: nodeId,
                        type: nodeType,
                        x: x,
                        y: y,
                        params: {}
                    };
                    
                    console.log('Adding new node:', newNode);
                    nodes.push(newNode);
                    renderNodes();
                    
                    // Streamlit에 업데이트 알림
                    if (window.parent && window.parent.postMessage) {
                        window.parent.postMessage({
                            type: 'update_network',
                            nodes: nodes,
                            edges: edges
                        }, '*');
                    }
                });
            });
            
            // 캔버스에 드롭
            document.getElementById('canvas').addEventListener('dragover', function(e) {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
            });
            
            document.getElementById('canvas').addEventListener('drop', function(e) {
                e.preventDefault();
                const nodeType = e.dataTransfer.getData('text/plain');
                console.log('Drop event:', nodeType);
                
                const rect = this.getBoundingClientRect();
                const x = e.clientX - rect.left - 60;
                const y = e.clientY - rect.top - 40;
                
                // 새 노드 생성
                const nodeId = nodeType.toLowerCase() + '_' + Date.now();
                const newNode = {
                    id: nodeId,
                    type: nodeType,
                    x: Math.max(0, Math.min(x, rect.width - 120)),
                    y: Math.max(0, Math.min(y, rect.height - 80)),
                    params: {}
                };
                
                console.log('Adding new node:', newNode);
                nodes.push(newNode);
                renderNodes();
                
                // Streamlit에 업데이트 알림
                if (window.parent && window.parent.postMessage) {
                    window.parent.postMessage({
                        type: 'update_network',
                        nodes: nodes,
                        edges: edges
                    }, '*');
                }
            });
            
            // 줌/팬 함수들
            function zoomIn() {
                zoomLevel = Math.min(zoomLevel * 1.2, 3);
                updateZoom();
            }
            
            function zoomOut() {
                zoomLevel = Math.max(zoomLevel / 1.2, 0.1);
                updateZoom();
            }
            
            function resetZoom() {
                zoomLevel = 1;
                panX = 0;
                panY = 0;
                updateZoom();
            }
            
            function updateZoom() {
                const canvas = document.getElementById('canvas');
                
                // 캔버스에만 transform 적용 (자식 요소들은 자동으로 상속)
                const transform = 'translate(' + panX + 'px, ' + panY + 'px) scale(' + zoomLevel + ')';
                canvas.style.transform = transform;
                canvas.style.transformOrigin = '0 0';
                
                document.getElementById('zoomLevel').textContent = Math.round(zoomLevel * 100) + '%';
            }
            
            function openPopup() {
                alert('팝업 창 기능은 현재 개발 중입니다. 곧 사용 가능합니다!');
            }
            
            // 컨트롤 함수들
            function clearCanvas() {
                if (confirm('모든 노드를 삭제하시겠습니까?')) {
                    nodes = [];
                    edges = [];
                    renderNodes();
                }
            }
            
            function deleteSelected() {
                if (selectedNode) {
                    const nodeId = selectedNode.dataset.nodeId;
                    nodes = nodes.filter(n => n.id !== nodeId);
                    edges = edges.filter(e => e[0] !== nodeId && e[1] !== nodeId);
                    renderNodes();
                    selectedNode = null;
                }
            }
            
            function generateCode() {
                // Streamlit에 데이터 전송
                if (window.parent && window.parent.postMessage) {
                    window.parent.postMessage({
                        type: 'update_network',
                        nodes: nodes,
                        edges: edges
                    }, '*');
                }
            }
            
            // 마우스 휠 줌 이벤트
            document.getElementById('canvasContainer').addEventListener('wheel', function(e) {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                zoomLevel = Math.max(0.1, Math.min(3, zoomLevel * delta));
                updateZoom();
            });
            
            // 팬 기능 (캔버스 드래그)
            document.getElementById('canvasContainer').addEventListener('mousedown', function(e) {
                if (e.target === this || e.target.classList.contains('canvas')) {
                    isPanning = true;
                    panStart.x = e.clientX - panX;
                    panStart.y = e.clientY - panY;
                    this.style.cursor = 'grabbing';
                }
            });
            
            document.addEventListener('mousemove', function(e) {
                if (isPanning) {
                    panX = e.clientX - panStart.x;
                    panY = e.clientY - panStart.y;
                    
                    // 캔버스에만 transform 적용 (자식 요소들은 자동으로 상속)
                    const canvas = document.getElementById('canvas');
                    const transform = 'translate(' + panX + 'px, ' + panY + 'px) scale(' + zoomLevel + ')';
                    
                    canvas.style.transform = transform;
                }
            });
            
            document.addEventListener('mouseup', function() {
                isPanning = false;
                document.getElementById('canvasContainer').style.cursor = 'grab';
            });
            
            // 초기 렌더링
            console.log('Initial nodes:', nodes);
            console.log('Initial edges:', edges);
            
            // DOM이 로드된 후 렌더링
            document.addEventListener('DOMContentLoaded', function() {
                console.log('DOM loaded, rendering nodes');
                renderNodes();
                updateZoom();
            });
            
            // 즉시 렌더링도 시도
            setTimeout(function() {
                renderNodes();
                updateZoom();
            }, 100);
            
            // 클릭으로 연결 생성 (간단한 버전)
            let connectionStart = null;
            document.getElementById('canvas').addEventListener('click', function(e) {
                const nodeEl = e.target.closest('.node');
                if (nodeEl) {
                    const nodeId = nodeEl.dataset.nodeId;
                    
                    if (!connectionStart) {
                        connectionStart = nodeId;
                        nodeEl.style.border = '3px solid #ffc107';
                    } else if (connectionStart !== nodeId) {
                        // 연결 생성
                        const newEdge = [connectionStart, nodeId];
                        if (!edges.some(e => e[0] === newEdge[0] && e[1] === newEdge[1])) {
                            edges.push(newEdge);
                            renderConnections();
                        }
                        
                        // 연결 모드 해제
                        document.querySelectorAll('.node').forEach(node => {
                            node.style.border = '2px solid transparent';
                        });
                        connectionStart = null;
                    }
                } else {
                    // 캔버스 클릭 시 연결 모드 해제
                    document.querySelectorAll('.node').forEach(node => {
                        node.style.border = '2px solid transparent';
                    });
                    connectionStart = null;
                }
            });
        </script>
    </body>
    </html>
    """
    
    return html

# ---------- 메인 UI ----------
st.title("🧠 TorchCanvas - 신경망 디자이너")

# 사이드바
with st.sidebar:
    st.header("🎨 템플릿")
    
    # 템플릿 선택
    selected_template = st.selectbox(
        "아키텍처 템플릿",
        ["빈 캔버스"] + list(TEMPLATES.keys()),
        help="기본 아키텍처 템플릿을 선택하세요"
    )
    
    if st.button("템플릿 로드", disabled=selected_template == "빈 캔버스"):
        if selected_template in TEMPLATES:
            template = TEMPLATES[selected_template]
            st.session_state.nodes = template["nodes"].copy()
            st.session_state.edges = template["edges"].copy()
            st.rerun()
    
    st.divider()
    
    st.header("📊 네트워크 정보")
    st.metric("노드 수", len(st.session_state.nodes))
    st.metric("연결 수", len(st.session_state.edges))
    
    st.divider()
    
    st.header("⚙️ 설정")
    if st.button("캔버스 초기화"):
        st.session_state.nodes = []
        st.session_state.edges = []
        st.session_state.selected_node = None
        st.rerun()

# 메인 영역
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎨 네트워크 캔버스")
    
    # 시각화 HTML 생성 및 표시
    viz_html = create_visualization_html(st.session_state.nodes, st.session_state.edges)
    
    # 디버깅 정보 표시
    st.write("🔍 디버깅 정보:")
    st.write(f"- 노드 수: {len(st.session_state.nodes)}")
    st.write(f"- 연결 수: {len(st.session_state.edges)}")
    if st.session_state.nodes:
        st.write("- 노드들:", [n['id'] for n in st.session_state.nodes])
    
    components.html(viz_html, height=850, scrolling=False)
    
    # JavaScript에서 데이터 업데이트 처리
    if st.session_state.get('network_update'):
        st.session_state.nodes = st.session_state.network_update.get('nodes', [])
        st.session_state.edges = st.session_state.network_update.get('edges', [])
        st.session_state.network_update = None
        st.rerun()

with col2:
    st.subheader("💻 생성된 코드")
    
    if st.session_state.nodes:
        # PyTorch 코드 생성
        pytorch_code = generate_pytorch_code(st.session_state.nodes, st.session_state.edges)
        
        # 코드 표시
        st.code(pytorch_code, language='python')
        
        # 코드 다운로드
        st.download_button(
            label="📥 코드 다운로드",
            data=pytorch_code,
            file_name="network.py",
            mime="text/python"
        )
        
        # 네트워크 정보
        st.subheader("📋 네트워크 구조")
        
        # 노드 정보
        for node in st.session_state.nodes:
            with st.expander(f"{NODE_TYPES[node['type']]['icon']} {node['id']} ({node['type']})"):
                st.write(f"**타입:** {node['type']}")
                st.write(f"**위치:** ({node['x']}, {node['y']})")
                if node.get('params'):
                    st.write("**파라미터:**")
                    for key, value in node['params'].items():
                        st.write(f"- {key}: {value}")
        
        # 연결 정보
        if st.session_state.edges:
            st.subheader("🔗 연결")
            for edge in st.session_state.edges:
                if len(edge) == 2:
                    st.write(f"`{edge[0]}` → `{edge[1]}`")
    else:
        st.info("캔버스에 노드를 추가하여 네트워크를 설계하세요!")
        st.markdown("""
        **사용 방법:**
        1. 왼쪽 팔레트에서 노드를 드래그하여 캔버스에 추가
        2. 노드를 클릭하여 이동
        3. 노드를 클릭하여 연결 생성
        4. 오른쪽에서 생성된 PyTorch 코드 확인
        """)

# JavaScript 메시지 처리 (개선된 버전)
if st.session_state.get('js_message'):
    message = st.session_state.js_message
    if message.get('type') == 'update_network':
        st.session_state.nodes = message.get('nodes', [])
        st.session_state.edges = message.get('edges', [])
        st.session_state.js_message = None
        st.rerun()

# JavaScript 메시지 리스너 추가 (개선된 버전)
js_listener = """
<script>
console.log('Message listener loaded');
window.addEventListener('message', function(event) {
    console.log('Received message:', event.data);
    if (event.data.type === 'update_network') {
        // Streamlit에 메시지 전송
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: event.data
        }, '*');
    }
});
</script>
"""

components.html(js_listener, height=0)
