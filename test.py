# TorchCanvas - Graph Compiler (Lazy version, A안 적용)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn

# ------------------------------
# 1) 레이어 레지스트리
# ------------------------------
class LayerFactory:
    registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name):
        def deco(fn):
            cls.registry[name] = fn
            return fn
        return deco

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls.registry:
            raise ValueError(f"Unknown layer: {name}")
        return cls.registry[name](**kwargs)

@LayerFactory.register("Input")
def _input(**kwargs):
    # placeholder: forward에서 입력을 그대로 통과
    return nn.Identity()

@LayerFactory.register("Conv1d")
def _conv1d(out_channels:int, kernel_size:int, stride:int=1, padding:str|int="same", **kwargs):
    pad = kernel_size // 2 if padding == "same" else padding
    return nn.LazyConv1d(out_channels, kernel_size, stride=stride, padding=pad, bias=True)

@LayerFactory.register("BatchNorm1d")
def _bn1d(num_features:int=None, **kwargs):
    # num_features 지정이 없으면 Lazy로
    return nn.LazyBatchNorm1d() if (num_features is None or num_features == 0) else nn.BatchNorm1d(num_features)

@LayerFactory.register("Linear")
def _linear(out_features:int, bias:bool=True, **kwargs):
    return nn.LazyLinear(out_features, bias=bias)

@LayerFactory.register("ReLU")
def _relu(**kwargs):
    return nn.ReLU(inplace=True)

@LayerFactory.register("Dropout")
def _dropout(p: float = 0.5, **kwargs):
    return nn.Dropout(p)

@LayerFactory.register("MaxPool1d")
def _maxpool1d(kernel_size:int=2, stride:int=None, **kwargs):
    return nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

@LayerFactory.register("Concat")
def _concat(dim:int=1, **kwargs):
    class Concat(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
        def forward(self, *xs):
            return torch.cat(xs, dim=self.dim)
    return Concat(dim)

@LayerFactory.register("Permute_BCT_to_BTH")
def _p_bct_to_bth(**kwargs):
    class P(nn.Module):
        def forward(self, x):  # (B,C,T) -> (B,T,C)
            return x.transpose(1, 2).contiguous()
    return P()

@LayerFactory.register("Permute_BTH_to_BCT")
def _p_bth_to_bct(**kwargs):
    class P(nn.Module):
        def forward(self, x):  # (B,T,C) -> (B,C,T)
            return x.transpose(1, 2).contiguous()
    return P()

@LayerFactory.register("GRUBlock")
def _gru_block(hidden_size:int, num_layers:int=1, bidirectional:bool=True,
               out:str="last", **kwargs):
    """
    GRU + 출력 정규화 블록
    - 입력: (B, T, H_in)
    - out='last'|'mean'|'max'|'seq'
    """
    class GRUBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.out_mode = out
            self.gru: nn.GRU | None = None  # Lazy: 첫 forward에서 생성

        def forward(self, x):  # (B,T,H_in)
            if x.ndim != 3:
                raise ValueError(f"GRUBlock expects 3D (B,T,H), got {x.shape}")
            if self.gru is None:
                self.gru = nn.GRU(
                    input_size=x.size(-1),
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    bidirectional=self.bidirectional,
                    batch_first=True,
                ).to(x.device, x.dtype)
            y, _ = self.gru(x)  # (B,T,D)
            if self.out_mode == "seq":
                return y
            if self.out_mode == "last":
                return y[:, -1, :]              # (B,D)
            if self.out_mode == "mean":
                return y.mean(dim=1)            # (B,D)
            if self.out_mode == "max":
                return y.max(dim=1).values      # (B,D)
            raise ValueError(f"Unknown out mode: {self.out_mode}")
    return GRUBlock()

# ------------------------------
# 2) 그래프 스키마
# ------------------------------
@dataclass
class Node:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    src: str
    dst: str

@dataclass
class GraphDef:
    nodes: List[Node]
    edges: List[Edge]
    inputs: List[str]
    outputs: List[str]

# ------------------------------
# 3) 그래프 → Module 컴파일러
# ------------------------------
class GraphModule(nn.Module):
    def __init__(self, g: GraphDef):
        super().__init__()
        self.g = g
        self.node_by_id: Dict[str, Node] = {n.id: n for n in g.nodes}
        self.topo = self._toposort()

        # 인접/역인접
        self.in_edges: Dict[str, List[str]] = {n.id: [] for n in g.nodes}
        for e in g.edges:
            self.in_edges[e.dst].append(e.src)

        # 모듈 생성
        self.nodes = nn.ModuleDict({n.id: self._instantiate(n) for n in g.nodes})

    def _instantiate(self, n: Node) -> nn.Module:
        return LayerFactory.create(n.type, **n.params)

    def _toposort(self):
        indeg = {n.id: 0 for n in self.g.nodes}
        adj: Dict[str, List[str]] = {}
        for e in self.g.edges:
            indeg[e.dst] += 1
            adj.setdefault(e.src, []).append(e.dst)
        q = [nid for nid, d in indeg.items() if d == 0]
        order = []
        while q:
            u = q.pop()
            order.append(u)
            for v in adj.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) != len(self.g.nodes):
            raise ValueError("Graph has cycle or disconnected parts.")
        return order

    def forward(self, inputs: Dict[str, torch.Tensor]):
        cache: Dict[str, torch.Tensor] = {}
        # 입력 채우기
        for inp in self.g.inputs:
            if inp not in inputs:
                raise KeyError(f"Missing input tensor for node '{inp}'")
            cache[inp] = inputs[inp]

        # 토폴로지대로 실행
        for nid in self.topo:
            node_def = self.node_by_id[nid]
            typ = node_def.type
            if typ == "Input":
                continue

            srcs = self.in_edges[nid]
            xs = [cache[s] for s in srcs] if srcs else []
            mod = self.nodes[nid]

            if typ == "Concat":
                x = mod(*xs)
            else:
                # 단일 입력 가정(필요 시 멀티 입력 처리 추가)
                x_in = xs[0] if xs else None
                x = mod(x_in)
            cache[nid] = x

        return {oid: cache[oid] for oid in self.g.outputs}

# ------------------------------
# 4) 사용 예시
# ------------------------------
if __name__ == "__main__":
    # 예제 그래프: Input(B,C,T) → Conv1d → ReLU → Permute(B,T,C) → GRUBlock(out='last') → Linear → 출력
    g = GraphDef(
        nodes=[
            Node("inp", "Input"),
            Node("conv", "Conv1d", {"out_channels": 64, "kernel_size": 5, "padding": "same"}),
            Node("relu", "ReLU"),
            Node("perm", "Permute_BCT_to_BTH"),
            Node("gru",  "GRUBlock", {"hidden_size": 128, "out": "last"}),
            Node("fc",   "Linear", {"out_features": 18}),
        ],
        edges=[
            Edge("inp", "conv"),
            Edge("conv", "relu"),
            Edge("relu", "perm"),
            Edge("perm", "gru"),
            Edge("gru", "fc"),
        ],
        inputs=["inp"],
        outputs=["fc"],
    )

    model = GraphModule(g)
    x = torch.randn(4, 9, 256)          # (B,C,T)
    out = model({"inp": x})["fc"]       # (B, 18)
    print("out.shape =", out.shape)
