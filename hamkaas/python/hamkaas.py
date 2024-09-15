from abc import ABC, abstractmethod

from dataclasses import dataclass

from typing import List, Tuple, Dict, Optional, Union

import ctypes
import math
import torch
import torch.nn

# For now, we support only these types for hamkaas.
_SUPPORTED_TENSOR_TYPES = [
    torch.float32,
    torch.float64,
    torch.int64,
]

_MIN_TENSOR_DIMS = 1
_MAX_TENSOR_DIMS = 3


@dataclass
class TensorMeta:
    type: torch.dtype
    shape: List[int]

    def __init__(self, tensor: torch.Tensor):
        self.type = tensor.dtype
        self.shape = list(tensor.shape)

    def __init__(self, type: torch.dtype, shape: List[int]):
        self.type = type
        self.shape = shape


class HamKaasNode(ABC):
    @abstractmethod
    def get_type(self) -> torch.dtype:
        ...

    @abstractmethod
    def get_shape(self) -> List[int]:
        ...

    def eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        if id(self) in cache:
            return cache[id(self)]

        result = self.do_eval_slow(inputs, buffers, cache)

        cache[id(self)] = result

        return result

    @abstractmethod
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        ...

    def __add__(self, other):
        if isinstance(other, HamKaasNode):
            return SumNode(self, other)
        elif isinstance(other, torch.Tensor):
            return SumNode(self, ConstantTensor(other))
        
    def __mul__(self, other):
        if isinstance(other, HamKaasNode):
            return HadamardProductNode(self, other)
        elif isinstance(other, torch.Tensor):
            return HadamardProductNode(self, ConstantTensor(other))
        
    def __matmul__(self, other):
        if isinstance(other, HamKaasNode):
            return MatMulNode(self, other)
        elif isinstance(other, torch.Tensor):
            return MatMulNode(self, ConstantTensor(other))

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None:
                raise ValueError("Step is not supported for slicing")
            return SliceNode(self, index.start, index.stop)
        elif isinstance(index, int):
            return SliceNode(self, index, index + 1)
        else:
            raise ValueError("Unsupported index type")

    def replace(self, replacement, start, end):
        return ReplaceSlice(self, replacement, start, end)

    def reshape(self, shape: List[int]):
        return ReshapeNode(self, shape)
    
    def permute(self, permutation: List[int]):
        return Permute(self, permutation)

    def relu(self):
        return ReLUNode(self)
    
    def silu(self):
        return SiLUNode(self)
    
    def rms_norm(self, weights: "HamKaasNode"):
        return RMSNormNode(self, weights)

    def complex_hadamard_product(self, other: "HamKaasNode"):
        return ComplexHadamardProductNode(self, other)

    def sliced_softmax(self, prefix_size: int):
        return SlicedSoftmaxNode(self, prefix_size)


class InputTensor(HamKaasNode):
    def __init__(self, name: str, type: torch.dtype, shape: List[int]):
        super().__init__()

        if type not in _SUPPORTED_TENSOR_TYPES:
            raise ValueError(f"Unsupported tensor type: {type}")
        if len(shape) < _MIN_TENSOR_DIMS or len(shape) > _MAX_TENSOR_DIMS:
            raise ValueError(f"Unsupported tensor dimension: {len(shape)}")
        for dim in shape:
            if dim <= 0:
                raise ValueError(f"Invalid tensor shape: {shape}")

        self.name = name
        self.type = type
        self.shape = shape

    def get_type(self) -> torch.dtype:
        return self.type

    def get_shape(self) -> List[int]:
        return self.shape
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return inputs[self.name]

    def get_name(self) -> str:
        return self.name
    

class BufferTensor(HamKaasNode):
    def __init__(self, type: torch.dtype, shape: List[int]):
        super().__init__()

        if type not in _SUPPORTED_TENSOR_TYPES:
            raise ValueError(f"Unsupported tensor type: {type}")
        if len(shape) < _MIN_TENSOR_DIMS or len(shape) > _MAX_TENSOR_DIMS:
            raise ValueError(f"Unsupported tensor dimension: {len(shape)}")
        for dim in shape:
            if dim <= 0:
                raise ValueError(f"Invalid tensor shape: {shape}")

        self.type = type
        self.shape = shape

    def get_type(self) -> torch.dtype:
        return self.type

    def get_shape(self) -> List[int]:
        return self.shape
    
    def do_eval_slow(self, _: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return buffers[self.name]


class ConstantTensor(HamKaasNode):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()

        if tensor.dtype not in _SUPPORTED_TENSOR_TYPES:
            raise ValueError(f"Unsupported tensor type: {tensor.dtype}")
        if len(tensor.shape) < _MIN_TENSOR_DIMS or len(tensor.shape) > _MAX_TENSOR_DIMS:
            raise ValueError(f"Unsupported tensor dimension: {len(tensor.shape)}")
        if tensor.nelement() == 0:
            raise ValueError("Empty tensors are not supported")

        # We need tensors to be contiguous to pass them to C++ code.
        tensor = tensor.contiguous()

        self.tensor = tensor
        self.name = None

    def get_type(self) -> torch.dtype:
        return self.tensor.dtype

    def get_shape(self) -> List[int]:
        return list(self.tensor.shape)
    
    def do_eval_slow(self, _1: Dict[str, torch.Tensor], _2: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return self.tensor
    
    def get_name(self) -> str:
        return self.name
    
    def set_name(self, name: str) -> None:
        self.name = name

    def get_tensor(self) -> torch.Tensor:
        return self.tensor


class SumNode(HamKaasNode):
    def __init__(self, lhs: HamKaasNode, rhs: HamKaasNode):
        super().__init__()

        if lhs.get_type() != rhs.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        if lhs.get_type() not in [torch.float32, torch.float64]:
            raise ValueError("Sum is supported for float32 and float64 tensors only")

        if len(lhs.get_shape()) != len(rhs.get_shape()):
            raise ValueError(f"Shapes do not match for addition: {lhs.get_shape()} vs {rhs.get_shape()}")

        # If one of the rhs dimensions is 1 and the other is not, broadcast is done.
        for i in range(len(lhs.get_shape())):
            if lhs.get_shape()[i] != rhs.get_shape()[i] and rhs.get_shape()[i] != 1:
                raise ValueError(f"Shapes do not match for addition: {lhs.get_shape()} vs {rhs.get_shape()}")

        self.lhs = lhs
        self.rhs = rhs

    def get_type(self) -> torch.dtype:
        return self.lhs.get_type()
    
    def get_shape(self) -> List[int]:
        return self.lhs.get_shape()
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return self.lhs.eval_slow(inputs, buffers, cache) + self.rhs.eval_slow(inputs, buffers, cache)


class MatMulNode(HamKaasNode):
    def __init__(self, lhs: HamKaasNode, rhs: HamKaasNode):
        super().__init__()

        if lhs.get_type() != rhs.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        if lhs.get_type() not in [torch.float32, torch.float64]:
            raise ValueError("Matrix multiplication is supported for float32 and float64 tensors only")

        lhs_shape = lhs.get_shape()
        if len(lhs_shape) == 1:
            lhs_shape = [1, lhs_shape[0]]
        elif len(lhs_shape) == 3:
            if len(rhs.get_shape()) != 3:
                raise ValueError("Incompatible shapes for multiplication {lhs.get_shape()} vs {rhs.get_shape()}")
            if lhs_shape[0] != rhs.get_shape()[0]:
                raise ValueError("Incompatible shapes for multiplication {lhs.get_shape()} vs {rhs.get_shape()}")
            if lhs_shape[2] != rhs.get_shape()[1]:
                raise ValueError("Incompatible shapes for multiplication {lhs.get_shape()} vs {rhs.get_shape()}")
        elif len(lhs_shape) != 2 or len(rhs.get_shape()) != 2:
            raise ValueError("Only matrices are supported for multiplication")
        else:
            assert len(lhs_shape) == 2 and len(rhs.get_shape()) == 2
            if lhs_shape[1] != rhs.get_shape()[0]:
                raise ValueError(f"Shapes do not match for multiplication: {lhs.get_shape()} vs {rhs.get_shape()}")
        
        self.lhs = lhs
        self.rhs = rhs

    def get_type(self) -> torch.dtype:
        return self.lhs.get_type()
    
    def get_shape(self) -> List[int]:
        if len(self.lhs.get_shape()) == 1:
            return [self.rhs.get_shape()[1]]
        elif len(self.lhs.get_shape()) == 2:
            return [self.lhs.get_shape()[0], self.rhs.get_shape()[1]]
        elif len(self.lhs.get_shape()) == 3:
            return [self.lhs.get_shape()[0], self.lhs.get_shape()[1], self.rhs.get_shape()[2]]

    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return torch.matmul(self.lhs.eval_slow(inputs, buffers, cache), self.rhs.eval_slow(inputs, buffers, cache))


class ReLUNode(HamKaasNode):
    def __init__(self, input: HamKaasNode):
        super().__init__()

        if input.get_type() not in [torch.float32, torch.float64]:
            raise ValueError("Only float32 and float64 tensors are supported for ReLU")

        self.input = input

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.input.get_shape()

    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return self.input.eval_slow(inputs, buffers, cache).clamp(min=0)    


class SiLUNode(HamKaasNode):
    def __init__(self, input: HamKaasNode):
        super().__init__()

        if input.get_type() not in [torch.float32, torch.float64]:
            raise ValueError("Only float32 and float64 tensors are supported for SiLU")

        self.input = input

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.input.get_shape()
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        f = torch.nn.SiLU()
        return f(self.input.eval_slow(inputs, buffers, cache))


class SliceNode(HamKaasNode):
    def __init__(self, input: HamKaasNode, start: Optional[int] = None, end: Optional[int] = None):
        super().__init__()

        if input.get_type() not in [torch.float32, torch.float64]:
            raise ValueError("Only float32 and float64 tensors are supported for slicing")

        self.input = input

        if start is None:
            start = 0
        if start >= input.get_shape()[0]:
            raise ValueError(f"Start index {start} is out of bounds for shape {input.get_shape()}")

        self.start = start

        if end is None:
            end = input.get_shape()[0]
        if end > input.get_shape()[0]:
            raise ValueError(f"End index {end} is out of bounds for shape {input.get_shape()}")
        if start >= end:
            raise ValueError(f"Start index {start} is greater or equal to end index {end}")

        self.end = end

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return [self.end - self.start] + self.input.get_shape()[1:]
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return self.input.eval_slow(inputs, buffers, cache)[self.start:self.end]
  

class RMSNormNode(HamKaasNode):
    def __init__(self, input: HamKaasNode, weights: HamKaasNode):
        super().__init__()

        if input.get_type() != weights.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        if len(input.get_shape()) != 1 or len(weights.get_shape()) != 1:
            raise ValueError("Only vectors are supported for RMSNorm")
        if input.get_shape()[0] != weights.get_shape()[0]:
            raise ValueError(f"Shapes do not match for RMSNorm: {input.get_shape()} vs {weights.get_shape()}")

        self.input = input
        self.weights = weights

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.input.get_shape()

    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.input.eval_slow(inputs, buffers, cache)
        weight = self.weights.eval_slow(inputs, buffers, cache)
        out = torch.zeros_like(x)
        size = len(x)
        # calculate sum of squares
        ss = 0.0
        for j in range(size):
            ss += x[j] * x[j]
        ss /= size
        ss += 1e-5
        ss = 1.0 / math.sqrt(ss)
        # normalize and scale
        for j in range(size):
            out[j] = weight[j] * (ss * x[j])
        return out

class ReshapeNode(HamKaasNode):
    def __init__(self, input: HamKaasNode, shape: List[int]):
        super().__init__()

        if input.get_type() not in [torch.float32, torch.float64]:
            raise ValueError("Only float32 and float64 tensors are supported for reshape")

        lhs_elements = 1
        for i in range(len(input.get_shape())):
            lhs_elements *= input.get_shape()[i]
        result_elements = 1
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise ValueError(f"Invalid shape: {shape}")
            result_elements *= shape[i]
        if lhs_elements != result_elements:
            raise ValueError(f"Reshape tensor size mismatch: {lhs_elements} vs {result_elements}")

        self.input = input
        self.shape = shape

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.shape
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return self.input.eval_slow(inputs, buffers, cache).reshape(self.shape)
    

class ComplexHadamardProductNode(HamKaasNode):
    def __init__(self, lhs: HamKaasNode, rhs: HamKaasNode):
        super().__init__()

        if lhs.get_type() != rhs.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        if len(lhs.get_shape()) != len(rhs.get_shape()):
            raise ValueError("Shapes do not match for complex Hadamard product")
        if lhs.get_shape()[-1] != 2 or rhs.get_shape()[-1] != 2:
            raise ValueError("Complex dot product requires complex vectors")
        for i in range(len(lhs.get_shape()) - 1):
            if lhs.get_shape()[i] != rhs.get_shape()[i] and rhs.get_shape()[i] != 1:
                raise ValueError("Shapes do not match for complex Hadamard product")

        self.lhs = lhs
        self.rhs = rhs

    def get_type(self) -> torch.dtype:
        return self.lhs.get_type()
    
    def get_shape(self) -> List[int]:
        return self.lhs.get_shape()
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.lhs.eval_slow(inputs, buffers, cache)
        y = self.rhs.eval_slow(inputs, buffers, cache)
        out = torch.zeros_like(x)
        for j in range(len(x)):
            out[j][0] = x[j][0] * y[j][0] - x[j][1] * y[j][1]
            out[j][1] = x[j][0] * y[j][1] + x[j][1] * y[j][0]
        return out


class HadamardProductNode(HamKaasNode):
    def __init__(self, lhs: HamKaasNode, rhs: HamKaasNode):
        super().__init__()

        if lhs.get_type() != rhs.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        if len(lhs.get_shape()) != len(rhs.get_shape()):
            raise ValueError("Shapes do not match for Hadamard product")
        for i in range(len(lhs.get_shape())):
            if lhs.get_shape()[i] != rhs.get_shape()[i] and rhs.get_shape()[i] != 1:
                raise ValueError("Shapes do not match for Hadamard product")

        self.lhs = lhs
        self.rhs = rhs

    def get_type(self) -> torch.dtype:
        return self.lhs.get_type()
    
    def get_shape(self) -> List[int]:
        return self.lhs.get_shape()
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return self.lhs.eval_slow(inputs, buffers, cache) * self.rhs.eval_slow(inputs, buffers, cache)
    

class Permute(HamKaasNode):
    def __init__(self, input: HamKaasNode, permutation: List[int]):
        super().__init__()

        if input.get_type() not in [torch.float32, torch.float64]:
            raise ValueError("Only float32 and float64 tensors are supported for permutation")
        if len(input.get_shape()) != len(permutation):
            raise ValueError("Permutation must have the same length as the input shape")
        if set(permutation) != set(range(len(input.get_shape()))):
            raise ValueError("Permutation must be a permutation of the input shape")

        self.input = input
        self.permutation = permutation

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return [self.input.get_shape()[i] for i in self.permutation]
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return self.input.eval_slow(inputs, buffers, cache).permute(self.permutation)


class ReplaceSlice(HamKaasNode):
    def __init__(self, input: HamKaasNode, replacement: HamKaasNode, start: Union[HamKaasNode, int], end: Union[HamKaasNode, int]):
        super().__init__()

        if isinstance(start, int):
            start = ConstantTensor(torch.tensor([start], dtype=torch.int64))
        if isinstance(end, int):
            end = ConstantTensor(torch.tensor([end], dtype=torch.int64))

        if input.get_type() != replacement.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        if len(input.get_shape()) != 1 or len(replacement.get_shape()) != 1:
            raise ValueError("Only vectors are supported for replace")
        if start.get_type() != torch.int64 or end.get_type() != torch.int64:
            raise ValueError("Start and end indices must be of type int64")
        if start.get_shape() != [1] or end.get_shape() != [1]:
            raise ValueError("Start and end indices must be scalars")

        self.input = input
        self.replacement = replacement
        self.start = start
        self.end = end

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.input.get_shape()
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.input.eval_slow(inputs, buffers, cache)
        y = self.replacement.eval_slow(inputs, buffers, cache)

        start = self.start.eval_slow(inputs, buffers, cache).item()
        end = self.end.eval_slow(inputs, buffers, cache).item()
        x[start:end] = y
        return x


class SlicedSoftmaxNode(HamKaasNode):
    def __init__(self, input: HamKaasNode, prefix_size: Union[int, HamKaasNode]):
        super().__init__()

        if input.get_type() not in [torch.float32, torch.float64]:
            raise ValueError("Only float32 and float64 tensors are supported for sliced softmax")
        if isinstance(prefix_size, int):
            prefix_size = ConstantTensor(torch.tensor([prefix_size], dtype=torch.int64))
        if prefix_size.get_type() != torch.int64:
            raise ValueError("Prefix size must be of type int64")
        if prefix_size.get_shape() != [1]:
            raise ValueError("Prefix size must be a scalar")
        
        self.prefix_size = prefix_size
        self.input = input

    def get_type(self) -> torch.dtype:
        return self.input.get_type()

    def get_shape(self) -> List[int]:
        return self.input.get_shape()
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.input.eval_slow(inputs, buffers, cache).clone()
        size = self.prefix_size.eval_slow(inputs, buffers, cache).item()
        xs = [x[i].item() for i in range(size)]
        # find max value (for numerical stability)
        max_val = xs[0]
        for i in range(1, size):
            if xs[i] > max_val:
                max_val = xs[i]
        # exp and sum
        exp_sum = 0.0
        for i in range(size):
            xs[i] = math.exp(xs[i] - max_val)
            exp_sum += xs[i]
        # normalize
        for i in range(size):
            x[i] = xs[i] / exp_sum
        return x

@dataclass
class TraversalResult:
    script: str
    constants: dict[str, torch.Tensor]
    inputs: dict[str, TensorMeta]
    output: TensorMeta

    def __init__(self):
        self.script = ""
        self.constants = {}
        self.inputs = {}
        self.output = None


def traverse_node(node: HamKaasNode) -> TraversalResult:
    result = TraversalResult()

    node_to_index = {}

    def run(node: HamKaasNode) -> int:
        global next_constant_tensor_id

        if id(node) in node_to_index:
            return node_to_index[id(node)]
       
        def register_node(expr: str) -> int:
            index = len(node_to_index) + 1
            node_to_index[id(node)] = index
            result.script += f"${index} = {expr};\n"
            return index

        if isinstance(node, InputTensor):
            node_type = str(node.get_type()).removeprefix("torch.")
            if node.get_name() in result.inputs:
                raise ValueError(f"Multiple input tensors with name {node.get_name()} were found")
            result.inputs[node.get_name()] = TensorMeta(node.get_type(), node.get_shape())

            return register_node(f"InputTensor({node.get_name()}, {node_type}, {node.get_shape()})")
        elif isinstance(node, ConstantTensor):
            if node.get_name():
                if node.get_name() in result.constants:
                    raise ValueError(f"Multiple constant tensors with name {node.get_name()} were found")
            else:
                next_constant_tensor_id = 0
                while "constant_" + str(next_constant_tensor_id) in result.constants:
                    next_constant_tensor_id += 1
                node.set_name("constant_" + str(next_constant_tensor_id))

            result.constants[node.get_name()] = node.get_tensor()
            node_type = str(node.get_type()).removeprefix("torch.")
            return register_node(f"ConstantTensor({node.get_name()}, {node_type}, {node.get_shape()})")
        elif isinstance(node, SumNode):
            return register_node(f"SumNode(${run(node.lhs)}, ${run(node.rhs)})")
        elif isinstance(node, MatMulNode):
            return register_node(f"MatMulNode(${run(node.lhs)}, ${run(node.rhs)})")
        elif isinstance(node, ReLUNode):
            return register_node(f"ReLUNode(${run(node.input)})")
        elif isinstance(node, SiLUNode):
            return register_node(f"SiLUNode(${run(node.input)})")
        elif isinstance(node, SliceNode):
            return register_node(f"SliceNode(${run(node.input)}, {node.start}, {node.end})")
        elif isinstance(node, RMSNormNode):
            return register_node(f"RMSNormNode(${run(node.input)}, ${run(node.weights)})")
        elif isinstance(node, ReshapeNode):
            return register_node(f"ReshapeNode(${run(node.input)}, {node.shape})")
        elif isinstance(node, ComplexHadamardProductNode):
            return register_node(f"ComplexHadamardProductNode(${run(node.lhs)}, ${run(node.rhs)})")
        elif isinstance(node, HadamardProductNode):
            return register_node(f"HadamardProductNode(${run(node.lhs)}, ${run(node.rhs)})")
        elif isinstance(node, Permute):
            return register_node(f"Permute(${run(node.input)}, {node.permutation})")
        elif isinstance(node, ReplaceSlice):
            return register_node(f"ReplaceSlice(${run(node.input)}, ${run(node.replacement)}, ${run(node.start)}, ${run(node.end)})")
        elif isinstance(node, SlicedSoftmaxNode):
            return register_node(f"SlicedSoftmaxNode(${run(node.input)}, ${run(node.prefix_size)})")
        elif isinstance(node, BufferTensor):
            node_type = str(node.get_type()).removeprefix("torch.")
            return register_node(f"BufferTensor({node_type}, {node.get_shape()})")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    output = run(node)
    result.script += f"result = ${output};\n"

    result.output = TensorMeta(node.get_type(), node.get_shape())

    return result


class HamKaasPlugin:
    class InitializationResult(ctypes.Structure):
        _fields_ = [
            ("handle", ctypes.c_void_p),
            ("error", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    class CompilationResult(ctypes.Structure):
        _fields_ = [
            ("model", ctypes.c_void_p),
            ("error", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    class NamedTensor(ctypes.Structure):
        _fields_ = [
            ("name", ctypes.c_char_p),
            ("data", ctypes.c_void_p),
        ]

    class CompilationOptions(ctypes.Structure):
        _fields_ = [
            ("use_gpu", ctypes.c_bool),
            ("use_cudnn", ctypes.c_bool),
        ]

    def __init__(self, hamkaas_path: str) -> None:
        self.lib = ctypes.CDLL(hamkaas_path)

        self.lib.HamKaasFreeErrorMessage.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        self.lib.HamKaasFreeErrorMessage.restype = None

        self.lib.HamKaasInitialize.argtypes = []
        self.lib.HamKaasInitialize.restype = HamKaasPlugin.InitializationResult

        self.lib.HamKaasFinalize.argtypes = [
            ctypes.c_void_p, # handle
        ]
        self.lib.HamKaasFinalize.restype = None

        self.lib.HamKaasInverseElements.restype = ctypes.POINTER(ctypes.c_ubyte)
        self.lib.HamKaasInverseElements.argtypes = [
            ctypes.POINTER(ctypes.c_float), # input
            ctypes.POINTER(ctypes.c_float), # output
            ctypes.c_int, # size
        ]

        self.lib.HamKaasCompileModel.argtypes = [
            ctypes.c_void_p, # handle
            HamKaasPlugin.CompilationOptions, # options
            ctypes.c_char_p, # scriptString
            ctypes.POINTER(HamKaasPlugin.NamedTensor), # constantTensors
            ctypes.c_int, # constantTensorCount
        ]
        self.lib.HamKaasCompileModel.restype = HamKaasPlugin.CompilationResult

        self.lib.HamKaasFreeModel.argtypes = [
            ctypes.c_void_p, # handle
            ctypes.c_void_p, # model
        ]
        self.lib.HamKaasFreeModel.restype = None

        self.lib.HamKaasEvaluateModel.argtypes = [
            ctypes.c_void_p, # handle
            ctypes.c_void_p, # model
            ctypes.POINTER(HamKaasPlugin.NamedTensor), # inputTensors
            ctypes.c_int, # inputTensorCount
            ctypes.c_void_p, # outputTensor
        ]
        self.lib.HamKaasEvaluateModel.restype = ctypes.POINTER(ctypes.c_ubyte)

        result = self.lib.HamKaasInitialize()
        if result.error:
            error = ctypes.string_at(result.error).decode()
            self.lib.HamKaasFreeErrorMessage(result.error)
            raise RuntimeError(error)
        self.handle = result.handle

    def __del__(self) -> None:
        if self.lib:
            self.lib.HamKaasFinalize(self.handle)

    def inverse_elements(self, input: torch.tensor) -> torch.tensor:
        if input.dtype != torch.float32:
            raise ValueError("Only float32 tensors are supported for inverse_elements")
        
        output = torch.zeros_like(input)
        input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_float))
        output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float))

        result_ptr = self.lib.HamKaasInverseElements(input_ptr, output_ptr, torch.numel(input))
        if result_ptr:
            error = ctypes.string_at(result_ptr).decode()
            self.lib.HamKaasFreeErrorMessage(result_ptr)
            raise RuntimeError(error)

        return output
    
    def compile_model(self, node: HamKaasNode, use_gpu=False, use_cudnn=False) -> "HamKaasModel":
        traveral_result = traverse_node(node)
        named_tensors = [
            HamKaasPlugin.NamedTensor(
                name.encode("utf-8"),
                tensor.data_ptr(),
            )
            for name, tensor in traveral_result.constants.items()
        ]

        options = HamKaasPlugin.CompilationOptions(use_gpu=use_gpu, use_cudnn=use_cudnn)

        compilation_result = self.lib.HamKaasCompileModel(
            self.handle,
            options,
            traveral_result.script.encode("utf-8"),
            (HamKaasPlugin.NamedTensor * len(named_tensors))(*named_tensors),
            len(named_tensors),
        )

        if compilation_result.error:
            error = ctypes.string_at(compilation_result.error).decode()
            self.lib.HamKaasFreeErrorMessage(compilation_result.error)
            raise RuntimeError(error)

        return HamKaasModel(self, compilation_result.model, traveral_result.inputs, traveral_result.output)


class HamKaasModel:
    def __init__(self, plugin: HamKaasPlugin, model_ptr: int, inputs: Dict[str, TensorMeta], output: TensorMeta) -> None:
        self._plugin = plugin
        self._model_ptr = model_ptr
        self._inputs = inputs
        self._output = output

    def __del__(self) -> None:
        self._plugin.lib.HamKaasFreeModel(self._plugin.handle, self._model_ptr)

    def evaluate(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        named_tensors = []

        for name, tensor in inputs.items():
            if name not in self._inputs:
                raise ValueError(f"Unknown input tensor: {name}")
            if self._inputs[name].type != tensor.dtype:
                raise ValueError(f"Input tensor {name} has wrong type: {tensor.dtype} vs {self._inputs[name].type}")
            if self._inputs[name].shape != list(tensor.shape):
                raise ValueError(f"Input tensor {name} has wrong shape: {tensor.shape} vs {self._inputs[name].shape}")

            named_tensors.append(HamKaasPlugin.NamedTensor(
                name.encode("utf-8"),
                tensor.data_ptr(),
            ))

        for name in self._inputs:
            if name not in inputs:
                raise ValueError(f"Missing input tensor: {name}")

        output_tensor = torch.zeros(self._output.shape, dtype=self._output.type)
        output_tensor_ptr = ctypes.cast(output_tensor.data_ptr(), ctypes.c_void_p)

        error_ptr = self._plugin.lib.HamKaasEvaluateModel(
            self._plugin.handle,
            self._model_ptr,
            (HamKaasPlugin.NamedTensor * len(named_tensors))(*named_tensors),
            len(named_tensors),
            output_tensor_ptr,
        )

        if error_ptr:
            error = ctypes.string_at(error_ptr).decode()
            self._plugin.lib.HamKaasFreeErrorMessage(error_ptr)
            raise RuntimeError(error)

        return output_tensor
