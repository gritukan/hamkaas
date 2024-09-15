import ctypes
import torch

class Inverser:
    def __init__(self, olejit_path):
        self._lib = ctypes.CDLL(olejit_path)
        self._lib.InverseElements.restype = ctypes.POINTER(ctypes.c_ubyte)

    def inverse(self, x):
        y = torch.zeros_like(x)
        x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))
        y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_float))
        error_ptr = self._lib.InverseElements(x_ptr, y_ptr, torch.numel(x))
        if error_ptr:
            error = ctypes.string_at(error_ptr).decode()
            self._lib.FreeErrorMessage(error_ptr)
            raise RuntimeError(error)
        return y


x = torch.tensor([[0, 2], [3, 4]], dtype=torch.float32)
inverser = Inverser("../cpp/libolejit.so")

print(inverser.inverse(x))
