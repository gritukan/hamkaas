import hamkaas
import torch

hamkaas.initialize("../cpp/libhamkaas.so")

c_1 = torch.tensor([
    [
        [1, 2, 3],
        [3, 4, 5]
    ],
    [
        [5, 7, -1],
        [3, 4, 1],
    ]
], dtype=torch.float32)

c_2 = torch.tensor([
    [
        [32, 49],
        [21, 19],
        [1, 1]
    ],
    [
        [5, 3],
        [-1, 7],
        [-2, 3]
    ]
], dtype=torch.float32)

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
