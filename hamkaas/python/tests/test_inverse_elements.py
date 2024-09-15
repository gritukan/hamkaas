import hamkaas

import pytest
import torch

class TestInverseElements:
    def setup_class(cls):
        cls.plugin = hamkaas.HamKaasPlugin("../../cpp/libhamkaas.so")

    def do_test(self, input):
        expected = 1.0 / input
        output = self.plugin.inverse_elements(input)
        assert torch.allclose(output, expected)

    def test_inverse_elements(self):
        self.do_test(torch.tensor([-1.0, 1.0, 2.0, 3.0]))
        self.do_test(torch.tensor([]))
        self.do_test(torch.tensor([1.0] * 100000))
        self.do_test(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_inverse_elements_error(self):
        # Wrong type.
        with pytest.raises(Exception):
            self.do_test(torch.tensor([1.0], dtype=torch.int32))

        # Division by zero.
        with pytest.raises(Exception):
            self.do_test(torch.tensor([0.0]))
        with pytest.raises(Exception):
            self.do_test(torch.tensor([-1.0, 0.0, 1.0]))
