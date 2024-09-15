from abc import ABC, abstractmethod

from dataclasses import dataclass

from typing import List, Tuple, Dict, Optional, Union

import ctypes
import math
import torch
import torch.nn

# For now, we support only these types for hamkaas.
_SUPPORTED_TENSOR_TYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
]

# For now, we support only vectors and matrices.
_MAX_TENSOR_DIMS = 2

class HamKaasNode(ABC):
    def __init__(self):
        self.debug = False
        pass

    def set_debug(self):
        self.debug = True

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
        if self.debug:
            tot = 1
            for i in range(len(result.shape)):
                tot *= result.shape[i]
            l = result.reshape([tot])
            l = [x.item() for x in result]

            print("new ", l[:3], l[len(l)//2:len(l)//2+3], l[-3:])

        cache[id(self)] = result

        return result

    @abstractmethod
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        ...


class InputTensor(HamKaasNode):
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
    
    def do_eval_slow(self, inputs: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return inputs[self.name]

    def get_name(self) -> str:
        return self.name
    

class BufferNode(HamKaasNode):
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
    
    def do_eval_slow(self, _: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        return buffers[self.name]


class ConstantTensor(HamKaasNode):
    def __init__(self, tensor: torch.Tensor, name: str = None):
        super().__init__()

        if tensor.dtype not in _SUPPORTED_TENSOR_TYPES:
            raise ValueError(f"Unsupported tensor type: {tensor.dtype}")
        if len(tensor.shape) > _MAX_TENSOR_DIMS:
            raise ValueError(f"Unsupported tensor dimension: {len(tensor.shape)}")

        # We need tensors to be contiguous to pass them to C++ code.
        tensor = tensor.contiguous()

        self.tensor = tensor
        self.name = name

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


class MulNode(HamKaasNode):
    def __init__(self, lhs: HamKaasNode, rhs: HamKaasNode):
        super().__init__()

        if lhs.get_type() != rhs.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        
        lhs_shape = lhs.get_shape()
        if len(lhs_shape) == 1:
            lhs_shape = [1, lhs_shape[0]]
        elif len(lhs_shape) == 3:
            if len(rhs.get_shape()) != 3:
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
        lhs = self.lhs.eval_slow(inputs, buffers, cache)
        rhs = self.rhs.eval_slow(inputs, buffers, cache)
        # if self.debug:
        #     print('lhs', lhs[0], lhs[1])
        #     import sys
        #     sys.exit(0)
        #     print('rhs', self.rhs.get_shape())
        return torch.matmul(self.lhs.eval_slow(inputs, buffers, cache), self.rhs.eval_slow(inputs, buffers, cache))


class ReLUNode(HamKaasNode):
    def __init__(self, input: HamKaasNode):
        super().__init__()

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

        lhs_elements = 1
        for i in range(len(input.get_shape())):
            lhs_elements *= input.get_shape()[i]
        result_elements = 1
        for i in range(len(shape)):
            result_elements *= shape[i]
        if lhs_elements != result_elements:
            raise ValueError(f"Reshape mismatch: {lhs_elements} vs {result_elements}")

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
        if len(lhs.get_shape()) != 2 or len(rhs.get_shape()) != 2:
            raise ValueError("Only vectors are supported for complex dot product")
        if lhs.get_shape()[1] != 2 or rhs.get_shape()[1] != 2:
            raise ValueError("Complex dot product requires complex vectors")
        if lhs.get_shape()[0] != rhs.get_shape()[0]:
            raise ValueError(f"Shapes do not match for complex Hadamard product: {lhs.get_shape()} vs {rhs.get_shape()}")

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
    def __init__(self, input: HamKaasNode, prefix_size: HamKaasNode):
        super().__init__()

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
class HamkaasScript:
    script: str
    constants: dict[str, torch.Tensor]

    def __init__(self):
        self.script = ""
        self.constants = {}


def create_script(node: HamKaasNode) -> HamkaasScript:
    result = HamkaasScript()

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
        elif isinstance(node, MulNode):
            return register_node(f"MulNode(${run(node.lhs)}, ${run(node.rhs)})")
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
            return register_node(f"ReplaceSlice(${run(node.input)}, ${run(node.replacement)}, {node.start}, {node.end})")
        elif isinstance(node, SlicedSoftmaxNode):
            return register_node(f"SlicedSoftmaxNode(${run(node.input)}, ${run(node.prefix_size)})")
        elif isinstance(node, BufferNode):
            node_type = str(node.get_type()).removeprefix("torch.")
            return register_node(f"BufferNode({node_type}, {node.get_shape()})")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    output = run(node)
    result.script += f"result = ${output};\n"

    return result


class HamKaasLibrary:
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

    def __init__(self, hamkaas_path: str) -> None:
        self.lib = ctypes.CDLL(hamkaas_path)

        self.lib.HamKaasCompileModel.argtypes = [
            ctypes.c_char_p, # scriptString
            ctypes.POINTER(HamKaasLibrary.NamedTensor), # constantTensors
            ctypes.c_int, # constantTensorCount
        ]
        self.lib.HamKaasCompileModel.restype = HamKaasLibrary.CompilationResult

        self.lib.HamKaasFreeModel.argtypes = [
            ctypes.c_void_p, # model
        ]
        self.lib.HamKaasFreeModel.restype = None

        self.lib.HamKaasEvaluateModel.argtypes = [
            ctypes.c_void_p, # model
            ctypes.POINTER(HamKaasLibrary.NamedTensor), # inputTensors
            ctypes.c_int, # inputTensorCount
            ctypes.c_void_p, # outputTensor
        ]
        self.lib.HamKaasEvaluateModel.restype = ctypes.POINTER(ctypes.c_ubyte)

_HAM_KAAS_LIBRARY = None

def initialize(ham_kaas_path: str) -> None:
    global _HAM_KAAS_LIBRARY
    _HAM_KAAS_LIBRARY = HamKaasLibrary(ham_kaas_path)

class HamKaasModel:
    def __init__(self, model_ptr: int, output_shape: List[int], output_type: torch.dtype) -> None:
        if _HAM_KAAS_LIBRARY is None:
            raise ValueError("Hamkaas library is not initialized")

        self._model_ptr = model_ptr
        self._output_shape = output_shape
        self._output_type = output_type

    def __del__(self) -> None:
        assert _HAM_KAAS_LIBRARY
        _HAM_KAAS_LIBRARY.lib.HamKaasFreeModel(self._model_ptr)

    def evaluate(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if _HAM_KAAS_LIBRARY is None:
            raise ValueError("Hamkaas library is not initialized")

        named_tensors = [
            HamKaasLibrary.NamedTensor(
                name.encode("utf-8"),
                tensor.data_ptr(),
            )
            for name, tensor in inputs.items()
        ]

        output_tensor = torch.zeros(self._output_shape, dtype=self._output_type)
        output_tensor_ptr = ctypes.cast(output_tensor.data_ptr(), ctypes.c_void_p)

        error_ptr = _HAM_KAAS_LIBRARY.lib.HamKaasEvaluateModel(
            self._model_ptr,
            (HamKaasLibrary.NamedTensor * len(named_tensors))(*named_tensors),
            len(named_tensors),
            output_tensor_ptr,
        )

        if error_ptr:
            error = ctypes.string_at(error_ptr).decode()
            _HAM_KAAS_LIBRARY.lib.HamKaasFreeErrorMessage(error_ptr)
            raise RuntimeError(error)

        return output_tensor

def compile_model(node: HamKaasNode) -> None:
    if _HAM_KAAS_LIBRARY is None:
        raise ValueError("Hamkaas library is not initialized")

    script = create_script(node)
    named_tensors = [
        HamKaasLibrary.NamedTensor(
            name.encode("utf-8"),
            tensor.data_ptr(),
        )
        for name, tensor in script.constants.items()
    ]

    compilation_result = _HAM_KAAS_LIBRARY.lib.HamKaasCompileModel(
        script.script.encode("utf-8"),
        (HamKaasLibrary.NamedTensor * len(named_tensors))(*named_tensors),
        len(named_tensors),
    )

    if compilation_result.error:
        print(compilation_result.error)
        error = ctypes.string_at(compilation_result.error).decode()
        _HAM_KAAS_LIBRARY.lib.HamKaasFreeErrorMessage(compilation_result.error)
        raise RuntimeError(error)

    return HamKaasModel(compilation_result.model, node.get_shape(), node.get_type())
