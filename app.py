# app.py — TorchCanvas Mini GUI (Graph → PyTorch 코드 미리보기/다운로드)
# 실행: streamlit run app.py
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any
import streamlit as st

st.set_page_config(page_title="TorchCanvas — Code Preview", layout="wide")

# ---------- 기본 노드 타입 스펙 (MVP) ----------
NODE_SPECS = {
    "Input": {"params": {}},
    "Conv1d": {"params": {"out_channels": int, "kernel_size": int, "stride": (int, 1), "padding": ("same_or_int", "same")}},
    "BatchNorm1d": {"params": {"num_features": (int, 0)}},  # 0 → LazyBatchNorm1d
    "Linear": {"params": {"out_features": int, "bias": (bool, True)}},
    "ReLU": {"params": {}},
    "Dropout": {"params": {"p": (float, 0.5)}},
    "MaxPool1d": {"params": {"kernel_size": (int, 2), "stride": (int, None)}},
    "Concat": {"params": {"dim": (int, 1)}},
    "Permute_BCT_to_BTH": {"params": {}},
    "Permute_BTH_to_BCT": {"params": {}},
    "GRUBlock": {"params": {
        "hidden_size": int, "num_layers": (int, 1), "bidirectional": (bool, True),
        "out": (["last","mean","max","seq"], "last")
    }},
}

# ---------- 세션 상태 초기화 ----------
def _init_state():
    ss = st.session_state
    ss.setdefault("nodes", [])    # type: ignore[assignment]  ← 필요시
    ss.setdefault("edges", [])
    ss.setdefault("inputs", [])
    ss.setdefault("outputs", [])
_init_state()

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
        elif typ == "BatchNorm1d":
            num = p.get("num_features", 0)
            if num in (None, 0):
                init_lines.append(f"self.{nid} = nn.LazyBatchNorm1d()")
            else:
                init_lines.append(f"self.{nid} = nn.BatchNorm1d({num})")
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
        elif typ == "GRUBlock":
            hs = p["hidden_size"]; nl = p.get("num_layers", 1); bd = p.get("bidirectional", True)
            outm = repr(p.get("out", "last"))
            init_lines.append(f"self.{nid} = GRUBlock(hidden_size={hs}, num_layers={nl}, bidirectional={bd}, out={outm})")
        elif typ in ("Concat","Permute_BCT_to_BTH","Permute_BTH_to_BCT"):
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
        elif typ == "Permute_BCT_to_BTH":
            fwd.append(f"cache['{nid}'] = cache['{srcs[0]}'].transpose(1, 2).contiguous()")
        elif typ == "Permute_BTH_to_BCT":
            fwd.append(f"cache['{nid}'] = cache['{srcs[0]}'].transpose(1, 2).contiguous()")
        elif typ in ("Conv1d","BatchNorm1d","Linear","ReLU","Dropout","MaxPool1d","GRUBlock"):
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
    x = torch.randn(4, 9, 256)  # (B,C,T)
    out = model({"inp": x})
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
    st.rerun()

# ---------- 메인: 탭 ----------
st.title("TorchCanvas — GUI → PyTorch 코드 미리보기")

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
    "version": "0.1",
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
        C = st.number_input("Channels(C)", min_value=1, value=9)
        T = st.number_input("Length(T)", min_value=1, value=256)
        run = st.button("Forward 실행")
        if run:
            # exec로 코드 실행해 ExportedModel 로드
            ns: Dict[str, Any] = {}
            try:
                exec(code_str, ns, ns)
                ExportedModel = ns["ExportedModel"]
                import torch
                m = ExportedModel()
                x = torch.randn(int(b), int(C), int(T))
                y = m({"inp": x})
                st.success(f"out.shape = {tuple(y.shape)}")
            except Exception as e:
                st.error(f"실행 에러: {e}")
