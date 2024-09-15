import torch
import hamkaas

input = hamkaas.InputTensor("in", torch.float32, [2, 2])
relu = hamkaas.ReLUNode(input)

res = hamkaas.create_script(relu)
print(res.script)
