import hamkaas
import torch

plugin = hamkaas.HamKaasPlugin("../cpp/libhamkaas.so")

in1_ = hamkaas.ConstantTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
out = in1_.sliced_softmax(2)
model = plugin.compile_model(out, use_gpu=True)

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
