# export_utils.py
import torch

# 람다 대신 래퍼 모듈로 trace/symbolic에 명확성 부여
class InferenceWrapper(torch.nn.Module):
    def __init__(self, gmod, input_key="inp", output_key="fc"):
        super().__init__()
        self.gmod = gmod
        self.input_key = input_key
        self.output_key = output_key
    def forward(self, x):
        return self.gmod({self.input_key: x})[self.output_key]

def to_torchscript(gmod, example, out_path="model.pt"):
    gmod.eval()
    with torch.no_grad():
        _ = gmod({"inp": example})  # lazy materialize
        wrapped = InferenceWrapper(gmod)
        scripted = torch.jit.trace(wrapped, example)
        scripted.save(out_path)

def to_onnx(gmod, example, out_path="model.onnx", opset=17):
    gmod.eval()
    with torch.no_grad():
        _ = gmod({"inp": example})  # lazy materialize
        wrapped = InferenceWrapper(gmod)
        torch.onnx.export(
            wrapped, example, out_path, opset_version=opset,
            input_names=["inp"], output_names=["fc"],
            dynamic_axes={"inp": {0: "B", 2: "T"}, "fc": {0: "B"}}
        )
