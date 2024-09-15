import hamkaas
import torch

hamkaas.initialize("../cpp/libhamkaas.so")

in_ = hamkaas.InputTensor("input", torch.float32, [5])
delta = hamkaas.ConstantTensor(torch.tensor([0.1], dtype=torch.float32))
out = hamkaas.SumNode(in_, delta)

model = hamkaas.compile_model(out)

input = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
print(model.evaluate({"input": input}))
