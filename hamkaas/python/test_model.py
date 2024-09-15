import hamkaas
import torch

hamkaas.initialize("../cpp/libhamkaas.so")

in1_ = hamkaas.ConstantTensor(torch.rand(3, 2, 10))
in2_ = hamkaas.ConstantTensor(torch.rand(3, 10, 5))
out = hamkaas.MulNode(in1_, in2_)
model = hamkaas.compile_model(out, use_gpu=True)

print(model.evaluate(inputs={}))
print(out.eval_slow({}, {}, {}))
#print(model.evaluate(inputs={}))
#print(model.evaluate(inputs={}))

"""
print(c_1 @ c_2)

c_1 = hamkaas.ConstantTensor(c_1)
c_2 = hamkaas.ConstantTensor(c_2)

out = hamkaas.MulNode(c_1, c_2)
model = hamkaas.compile_model(out, use_gpu=True)
print(model.evaluate(inputs={}))

import sys
sys.exit(0)

in1 = hamkaas.InputTensor("input", torch.int16, [2, 2])
in_ = hamkaas.InputTensor("input", torch.int16, [2, 2])
delta = hamkaas.ConstantTensor(torch.tensor([[1, 100]], dtype=torch.int16))
out = hamkaas.SumNode(in_, delta)

model = hamkaas.compile_model(out, use_gpu=True)

input = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)

print(model.evaluate({"input": input}))
"""
