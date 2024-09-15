from abc import ABC, abstractmethod

from dataclasses import dataclass

from typing import List, Tuple, Dict

import ctypes
import torch

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
        pass

    @abstractmethod
    def get_type(self) -> torch.dtype:
        ...

    @abstractmethod
    def get_shape(self) -> List[int]:
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

    def get_name(self) -> str:
        return self.name


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


class MulNode(HamKaasNode):
    def __init__(self, lhs: HamKaasNode, rhs: HamKaasNode):
        super().__init__()

        if lhs.get_type() != rhs.get_type():
            raise ValueError("Mixed-precision operations are not supported")
        
        if len(lhs.get_shape()) != 2 or len(rhs.get_shape()) != 2:
            print(lhs.get_shape())
            raise ValueError("Only matrices are supported for multiplication")
        
        if lhs.get_shape()[1] != rhs.get_shape()[0]:
            raise ValueError(f"Shapes do not match for multiplication: {lhs.get_shape()} vs {rhs.get_shape()}")
        
        self.lhs = lhs
        self.rhs = rhs

    def get_type(self) -> torch.dtype:
        return self.lhs.get_type()
    
    def get_shape(self) -> List[int]:
        return [self.lhs.get_shape()[0], self.rhs.get_shape()[1]]
    

class ReLUNode(HamKaasNode):
    def __init__(self, input: HamKaasNode):
        super().__init__()

        self.input = input

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.input.get_shape()
    

class SiLUNode(HamKaasNode):
    def __init__(self, input: HamKaasNode):
        super().__init__()

        self.input = input

    def get_type(self) -> torch.dtype:
        return self.input.get_type()
    
    def get_shape(self) -> List[int]:
        return self.input.get_shape()
    

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
            result.script += f"{index} = {expr};\n"
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
