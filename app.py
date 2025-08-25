# app.py — TorchCanvas Mini GUI (Graph → PyTorch 코드 미리보기/다운로드)
# 실행: streamlit run app.py
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="TorchCanvas — Code Preview", layout="wide")

# ---------- 기본 노드 타입 스펙 (MVP) ----------
NODE_SPECS = {
    "Input": {"params": {}},
    "Conv1d": {"params": {"out_channels": int, "kernel_size": int, "stride": (int, 1), "padding": ("same_or_int", "same")}},
    "Conv2d": {"params": {"out_channels": int, "kernel_size": int, "stride": (int, 1), "padding": ("same_or_int", "same")}},
    "BatchNorm1d": {"params": {"num_features": (int, 0)}},  # 0 → LazyBatchNorm1d
    "BatchNorm2d": {"params": {"num_features": (int, 0)}},  # 0 → LazyBatchNorm2d
    "Linear": {"params": {"out_features": int, "bias": (bool, True)}},
    "ReLU": {"params": {}},
    "Dropout": {"params": {"p": (float, 0.5)}},
    "MaxPool1d": {"params": {"kernel_size": (int, 2), "stride": (int, None)}},
    "MaxPool2d": {"params": {"kernel_size": (int, 2), "stride": (int, None)}},
    "Concat": {"params": {"dim": (int, 1)}},
    "Add": {"params": {}},
    "Flatten": {"params": {"start_dim": (int, 1), "end_dim": (int, -1)}},
    "Permute_BCT_to_BTH": {"params": {}},
    "Permute_BTH_to_BCT": {"params": {}},
    "GRUBlock": {"params": {
        "hidden_size": int, "num_layers": (int, 1), "bidirectional": (bool, True),
        "out": (["last","mean","max","seq"], "last")
    }},
    # 컴포지트 블록들
    "SEBlock": {"params": {"reduction": (int, 8)}},
    "ResidualBlock": {"params": {"out_channels": int, "kernel_size": (int, 3), "stride": (int, 1)}},
    "VGGBlock": {"params": {
        "c1": int, "c2": int, "kernel_size": (int, 3), 
        "use_lrn": (bool, False), "pool": (bool, True)
    }},
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

# ---------- 시각적 네트워크 다이어그램 생성 ----------
def create_network_diagram(nodes: List[dict], edges: List[List[str]], tensor_shapes: Dict[str, tuple]) -> str:
    """HTML/CSS로 네트워크 다이어그램 생성"""
    
    html = """
    <style>
    .network-container {
        font-family: 'Courier New', monospace;
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .layer {
        background: white;
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .layer-name {
        font-weight: bold;
        color: #007bff;
        margin-bottom: 5px;
    }
    .tensor-shape {
        font-family: 'Courier New', monospace;
        background: #e9ecef;
        padding: 5px 10px;
        border-radius: 4px;
        margin: 5px 0;
        font-size: 12px;
    }
    .status-check {
        position: absolute;
        top: 10px;
        right: 10px;
        color: #28a745;
        font-size: 18px;
    }
    .arrow {
        text-align: center;
        font-size: 20px;
        color: #6c757d;
        margin: 5px 0;
    }
    .params {
        font-size: 11px;
        color: #6c757d;
        margin-top: 5px;
    }
    </style>
    <div class="network-container">
    """
    
    # 노드들을 순서대로 정렬 (간단한 위상정렬)
    node_order = []
    in_edges = {n["id"]: [] for n in nodes}
    for src, dst in edges:
        in_edges[dst].append(src)
    
    # 입력 노드부터 시작
    for node in nodes:
        if node["id"] in st.session_state.inputs:
            node_order.append(node)
    
    # 나머지 노드들 추가 (간단한 방식)
    for node in nodes:
        if node not in node_order:
            node_order.append(node)
    
    for i, node in enumerate(node_order):
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
        
        html += f"""
        <div class="layer">
            <div class="status-check">✓</div>
            <div class="layer-name">{node_type}</div>
            <div class="tensor-shape">[{', '.join(map(str, output_shape))}]</div>
            {f'<div class="params">{param_str}</div>' if param_str else ''}
        </div>
        """
        
        if i < len(node_order) - 1:
            html += '<div class="arrow">↓</div>'
    
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
        elif typ in ("Conv1d","Conv2d","BatchNorm1d","BatchNorm2d","Linear","ReLU","Dropout","MaxPool1d","MaxPool2d","Flatten","GRUBlock","SEBlock","ResidualBlock","VGGBlock"):
            fwd.append(f"cache['{nid}'] = self.{nid}(cache['{srcs[0]}'])")
        else:
            fwd.append(f"# TODO: forward for '{typ}'")
    if len(outputs) == 1:
        fwd.append(f"return cache['{outputs[0]}']")
    else:
        pairs = ", ".join([f"'{k}': cache['{k}']" for k in outputs])
        fwd.append(f"return {{ {pairs} }}")
    fwd_body = "\n        ".join(fwd)

    header = """# Auto-generated by TorchCanvas Code Export
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUBlock(nn.Module):
    \"\"\"
    GRU + output normalization
    - input: (B, T, H_in)
    - out: 'last' | 'mean' | 'max' | 'seq'
    \"\"\"
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
    \"\"\"
    Squeeze-and-Excitation Block
    - input: (B, C, H, W) or (B, C, T)
    - reduction: channel reduction ratio
    \"\"\"
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
    \"\"\"
    Residual Block with optional projection
    - input: (B, C, H, W) or (B, C, T)
    - out_channels: output channels (if different from input, uses 1x1 projection)
    \"\"\"
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
    \"\"\"
    VGG-style block: Conv-ReLU-Conv-ReLU-(Pool)
    - input: (B, C, H, W) or (B, C, T)
    - c1, c2: channel counts for two conv layers
    \"\"\"
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
"""
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
    out = model({{"inp": x}})
    print("out.shape =", out.shape)
"""
    return header + cls + main

# ---------- 사이드바: 팔레트/노드 추가 ----------
st.sidebar.title("TorchCanvas — Palette")
with st.sidebar.form("add_node_form"):
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
    add = st.form_submit_button("노드 추가")
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

# ---------- 시각적 네트워크 다이어그램 생성 ----------
def create_network_diagram(nodes: List[dict], edges: List[List[str]], tensor_shapes: Dict[str, tuple]) -> str:
    """HTML/CSS로 네트워크 다이어그램 생성"""
    
    html = """
    <style>
    .network-container {
        font-family: 'Courier New', monospace;
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .layer {
        background: white;
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .layer-name {
        font-weight: bold;
        color: #007bff;
        margin-bottom: 5px;
    }
    .tensor-shape {
        font-family: 'Courier New', monospace;
        background: #e9ecef;
        padding: 5px 10px;
        border-radius: 4px;
        margin: 5px 0;
        font-size: 12px;
    }
    .status-check {
        position: absolute;
        top: 10px;
        right: 10px;
        color: #28a745;
        font-size: 18px;
    }
    .arrow {
        text-align: center;
        font-size: 20px;
        color: #6c757d;
        margin: 5px 0;
    }
    .params {
        font-size: 11px;
        color: #6c757d;
        margin-top: 5px;
    }
    </style>
    <div class="network-container">
    """
    
    # 노드들을 순서대로 정렬 (간단한 위상정렬)
    node_order = []
    in_edges = {n["id"]: [] for n in nodes}
    for src, dst in edges:
        in_edges[dst].append(src)
    
    # 입력 노드부터 시작
    for node in nodes:
        if node["id"] in st.session_state.inputs:
            node_order.append(node)
    
    # 나머지 노드들 추가 (간단한 방식)
    for node in nodes:
        if node not in node_order:
            node_order.append(node)
    
    for i, node in enumerate(node_order):
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
            # 간단한 형태 추정
            if not input_shapes:
                output_shape = (1, 3, 224, 224)
            else:
                output_shape = input_shapes[0]  # 기본적으로 입력 형태 유지
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
            param_items = param_items[:3]  # 최대 3개만 표시
            param_str = ", ".join(param_items)
        
        html += f"""
        <div class="layer">
            <div class="status-check">✓</div>
            <div class="layer-name">{node_type}</div>
            <div class="tensor-shape">[{', '.join(map(str, output_shape))}]</div>
            {f'<div class="params">{param_str}</div>' if param_str else ''}
        </div>
        """
        
        if i < len(node_order) - 1:
            html += '<div class="arrow">↓</div>'
    
    html += "</div>"
    return html

# ---------- 메인: 시각적 네트워크 다이어그램 ----------
st.title("TorchCanvas — 시각적 신경망 디자이너")

# 네트워크 다이어그램 표시
if st.session_state.nodes:
    st.subheader("🔍 네트워크 아키텍처 시각화")
    diagram_html = create_network_diagram(
        st.session_state.nodes, 
        st.session_state.edges, 
        st.session_state.tensor_shapes
    )
    st.components.v1.html(diagram_html, height=400, scrolling=True)
else:
    st.info("노드를 추가하여 네트워크를 구성하세요.")

# ---------- 텍스트 기반 노드/엣지 편집 ----------
colA, colB = st.columns([1,1], gap="large")
with colA:
    st.subheader("Nodes")
    st.write(st.session_state.nodes if st.session_state.nodes else "아직 노드가 없습니다.")
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
    st.subheader("Edges")
    st.write(st.session_state.edges if st.session_state.edges else "아직 엣지가 없습니다.")
    if st.session_state.edges:
        idx = st.number_input("삭제할 엣지 index", min_value=0, max_value=max(0,len(st.session_state.edges)-1), value=0, step=1)
        if st.button("엣지 삭제"):
            st.session_state.edges.pop(idx); st.rerun()

st.divider()
st.subheader("Graph JSON")
graph = {
    "version": "0.2",
    "metadata": {"name": "torchcanvas_ui"},
    "nodes": st.session_state.nodes,
    "edges": st.session_state.edges,
    "inputs": st.session_state.inputs,
    "outputs": st.session_state.outputs,
}
st.code(json.dumps(graph, indent=2, ensure_ascii=False), language="json")

# ---------- 코드 미리보기 / 다운로드 / 스모크 테스트 ----------
st.divider()
st.subheader("Generated PyTorch Code Preview")
code_str = ""
err = None
try:
    if st.session_state.nodes and st.session_state.inputs and st.session_state.outputs:
        code_str = export_graph_to_python(graph, class_name="ExportedModel")
        st.code(code_str, language="python")
    else:
        st.info("노드/엣지/입출력을 먼저 구성하세요.")
except Exception as e:
    err = str(e); st.error(err)

col1, col2 = st.columns([1,1])
with col1:
    if code_str:
        st.download_button("exported_model.py 다운로드", data=code_str.encode("utf-8"),
                           file_name="exported_model.py", mime="text/x-python")
with col2:
    if code_str:
        st.caption("⚙️ 스모크 테스트 (임의 텐서)")
        b = st.number_input("Batch", min_value=1, value=4)
        C = st.number_input("Channels(C)", min_value=1, value=3)
        H = st.number_input("Height(H)", min_value=1, value=224)
        W = st.number_input("Width(W)", min_value=1, value=224)
        run = st.button("Forward 실행")
        if run:
            # exec로 코드 실행해 ExportedModel 로드
            ns: Dict[str, Any] = {}
            try:
                exec(code_str, ns, ns)
                ExportedModel = ns["ExportedModel"]
                import torch
                m = ExportedModel()
                x = torch.randn(int(b), int(C), int(H), int(W))
                y = m({"inp": x})
                st.success(f"out.shape = {tuple(y.shape)}")
            except Exception as e:
                st.error(f"실행 에러: {e}")

# ---------- 템플릿 예시 ----------
st.divider()
st.subheader("템플릿 예시")

if st.button("VGG-16 템플릿 로드"):
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
    st.rerun()

if st.button("ResNet-18 템플릿 로드"):
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
    st.rerun()
