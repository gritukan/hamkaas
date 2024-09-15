import hamkaas
import torch

plugin = hamkaas.HamKaasPlugin("../cpp/libhamkaas.so")

t1 = hamkaas.ConstantTensor(torch.tensor([1, 2, 3, 4], dtype=torch.float32))
t2 = hamkaas.ConstantTensor(torch.tensor([5, 6], dtype=torch.float32))
x = t1.replace(t2, 1, 3)
# t1 = torch.rand((1, 8, 2))
# t2 = torch.rand((1, 2, 1))
# in1_ = hamkaas.ConstantTensor(t1)
# in2_ = hamkaas.ConstantTensor(t2)
# out = in1_ @ in2_
model = plugin.compile_model(x, use_gpu=True, use_cudnn=False)

for _ in range(1):
    print(model.evaluate(inputs={}))

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
