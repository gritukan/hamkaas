import torch
import olejit

input = olejit.InputTensor("in", torch.float32, [2, 2])
relu = olejit.ReLUNode(input)

res = olejit.create_script(relu)
print(res.script)
