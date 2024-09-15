from abc import ABC, abstractmethod

from dataclasses import dataclass

from typing import List, Tuple

import torch

# For now, we support only these types for olejit.
# The only exception is the output tensor that can be of integer type also.
_SUPPORTED_TENSOR_TYPES = [torch.float16, torch.float32, torch.float64]

# For now, we support only vectors and matrices.
_MAX_TENSOR_DIMS = 2

class OlejitNode(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_type(self) -> torch.dtype:
        ...

    @abstractmethod
    def get_shape(self) -> List[int]:
        ...


class InputTensor(OlejitNode):
    def __init__(self, name: str, type: torch.dtype, shape: List[int]):
        super().__init__()

        if type not in _SUPPORTED_TENSOR_TYPES:
            raise ValueError(f"Unsupported tensor type: {type}")
        if len(shape) > _MAX_TENSOR_DIMS:
            raise ValueError(f"Unsupported tensor dimension: {len(shape)}")

        self.name = name
        self.type = type
        self.shape = shape

    def get_type(self) -> torch.dtype:
        return self.type

    def get_shape(self) -> List[int]:
        return self.shape

    def get_name(self) -> str:
        return self.name


class ConstantTensor(OlejitNode):
    def __init__(self, tensor: torch.Tensor, name: str = None):
        super().__init__()

        if tensor.dtype not in _SUPPORTED_TENSOR_TYPES:
            raise ValueError(f"Unsupported tensor type: {tensor.dtype}")
        if len(tensor.shape) > _MAX_TENSOR_DIMS:
            raise ValueError(f"Unsupported tensor dimension: {len(tensor.shape)}")

        # We need tensors to be contiguous to pass them to C++ code.
        tensor = tensor.contiguous()

        self.tensor = tensor

    def get_type(self) -> torch.dtype:
        return self.tensor.dtype

    def get_shape(self) -> List[int]:
        return self.tensor.shape
    
    def get_name(self) -> str:
        return self.name
    
    def set_name(self, name: str) -> None:
        self.name = name

    def get_tensor(self) -> torch.Tensor:
        return self.tensor


class SumNode(OlejitNode):
    def __init__(self, lhs: OlejitNode, rhs: OlejitNode):
        super().__init__()

        if lhs.get_type() != rhs.get_type():
            raise ValueError("Mixed-precision operations are not supported")

        # If one of the rhs dimensions is 1 and the other is not, broadcast is done.
        for i in range(len(lhs.get_shape())):
            if lhs.get_shape()[i] != rhs.get_shape()[i] and lhs.get_shape()[i] != 1:
                raise ValueError(f"Shapes do not match for addition: {lhs.get_shape()} vs {rhs.get_shape()}")

        self.lhs = lhs
        self.rhs = rhs

    def get_type(self) -> torch.dtype:
        return self.lhs.get_type()
    
    def get_shape(self) -> List[int]:
        return self.lhs.get_shape()


class MulNode(OlejitNode):
    def __init__(self, lhs: OlejitNode, rhs: OlejitNode):
        super().__init__()

        if lhs.get_type() != rhs.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        
        if len(lhs.get_shape()) != 2 or len(rhs.get_shape()) != 2:
            raise ValueError("Only matrices are supported for multiplication")
        
        if lhs.get_shape()[1] != rhs.get_shape()[0]:
            raise ValueError(f"Shapes do not match for multiplication: {lhs.get_shape()} vs {rhs.get_shape()}")

    def get_type(self) -> torch.dtype:
        return self.lhs.get_type()
    
    def get_shape(self) -> List[int]:
        return [self.lhs.get_shape()[0], self.rhs.get_shape()[1]]
    

class ReLUNode(OlejitNode):
    def __init__(self, input: OlejitNode):
        super().__init__()

        self.input = input

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.input.get_shape()
    

class SiLUNode(OlejitNode):
    def __init__(self, input: OlejitNode):
        super().__init__()

        self.input = input

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.input.get_shape()
    

@dataclass
class OlejitScript:
    script: str
    constants: dict[str, torch.Tensor]

    def __init__(self):
        self.script = ""
        self.constants = {}


def create_script(node: OlejitNode) -> OlejitScript:
    result = OlejitScript()

    node_to_index = {}

    next_constant_tensor_id = 0

    def run(node: OlejitNode) -> int:
        if id(node) in node_to_index:
            return node_to_index[id(node)]

        def get_index():
            index = len(node_to_index) + 1
            node_to_index[id(node)] = index
            return index
        
        def register_node(expr: str) -> int:
            index = len(node_to_index) + 1
            node_to_index[id(node)] = index
            result.script += f"{index} = {expr};\n"
            return index

        if isinstance(node, InputTensor):
            return register_node(f"InputTensor({node.get_name()}, {node.get_type()}, {node.get_shape()})")
        elif isinstance(node, ConstantTensor):
            if node.get_name():
                if node.get_name() in result.constants:
                    raise ValueError(f"Multiple constant tensors with name {node.get_name()} were found")
            else:
                while "constant_" + str(next_constant_tensor_id) in result.constants:
                    next_constant_tensor_id += 1
                node.set_name("constant_" + str(next_constant_tensor_id))

            result.constants[node.get_name()] = node.get_tensor()
            return register_node(f"ConstantTensor({node.get_name()}, {node.get_type()}, {node.get_shape()})")
        elif isinstance(node, SumNode):
            lhs_index = run(node.lhs)
            rhs_index = run(node.rhs)
            return register_node(f"SumNode({lhs_index}, {rhs_index})")
        elif isinstance(node, MulNode):
            lhs_index = run(node.lhs)
            rhs_index = run(node.rhs)
            return register_node(f"MulNode({lhs_index}, {rhs_index})")
        elif isinstance(node, ReLUNode):
            input_index = run(node.input)
            return register_node(f"ReLUNode({input_index})")
        elif isinstance(node, SiLUNode):
            input_index = run(node.input)
            return register_node(f"SiLUNode({input_index})")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    output = run(node)
    result.script += f"result = {output};\n"

    return result
