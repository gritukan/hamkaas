import hamkaas

import math
import pytest
import torch


class TestCompilerCpu:
    USE_GPU = False
    USE_CUDNN = False

    def setup_class(cls):
        cls.plugin = hamkaas.HamKaasPlugin("../../cpp/libhamkaas.so")

    def compile(self, node):
        return self.plugin.compile_model(node, use_gpu=self.USE_GPU)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
    def test_constant_node(self, dtype):
        def do_test(tensor):
            node = hamkaas.ConstantTensor(tensor)
            model = self.compile(node)
            assert torch.allclose(model.evaluate({}), tensor)

        do_test(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
        do_test(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype))
        do_test(torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=dtype))
        do_test(torch.tensor([1.0] * 100000, dtype=dtype))

        # 0-d tensors are not supported.
        with pytest.raises(Exception):
            do_test(torch.tensor(1.0, dtype=dtype))

        # 4-d tensor are not supported.
        with pytest.raises(Exception):
            do_test(torch.tensor([[[[1.0]]]], dtype=dtype))
        
        # int32 tensors are not supported.
        with pytest.raises(Exception):
            do_test(torch.tensor([1.0], dtype=torch.int32))

        # Empty tensors are not supported.
        with pytest.raises(Exception):
            do_test(torch.tensor([], dtype=dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
    def test_input_node(self, dtype):
        def do_test(tensor):
            node = hamkaas.InputTensor("input", tensor.dtype, list(tensor.shape))
            model = self.compile(node)
            assert torch.allclose(model.evaluate({"input": tensor}), tensor)

        do_test(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
        do_test(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype))
        do_test(torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=dtype))
        do_test(torch.tensor([1.0] * 100000, dtype=dtype))

        # 0-d tensors are not supported.
        with pytest.raises(Exception):
            do_test(torch.tensor(1.0, dtype=dtype))

        # 4-d tensor are not supported.
        with pytest.raises(Exception):
            do_test(torch.tensor([[[[1.0]]]], dtype=dtype))
        
        # int32 tensors are not supported.
        with pytest.raises(Exception):
            do_test(torch.tensor([1.0], dtype=torch.int32))

        # Empty tensors are not supported.
        with pytest.raises(Exception):
            do_test(torch.tensor([], dtype=dtype))

        # Something weird.
        with pytest.raises(Exception):
            node = hamkaas.InputTensor("input", dtype, [2, 0, 2])
            self.compile(node)
        with pytest.raises(Exception):
            node = hamkaas.InputTensor("input", dtype, [1, -2, 3])
            self.compile(node)

    def test_invalid_input(self):
        input = hamkaas.InputTensor("input", torch.float32, [2, 2])
        model = self.compile(input)

        # Missing input.
        with pytest.raises(Exception):
            model.evaluate({})

        # Extra input.
        with pytest.raises(Exception):
            t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            model.evaluate({"input": t, "extra": t})

        # Wrong shape.
        with pytest.raises(Exception):
            model.evaluate({"input": torch.tensor([1.0, 2.0, 3.0])})

        # Wrong shape.
        with pytest.raises(Exception):
            model.evaluate({"input": torch.tensor([[1.0]])})

        # Wrong shape.
        with pytest.raises(Exception):
            model.evaluate({"input": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})

        # Wrong type.
        with pytest.raises(Exception):
            model.evaluate({"input": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)})

        # Two nodes with the same name.
        input2 = hamkaas.InputTensor("input", torch.float32, [2, 2])
        sum = hamkaas.SumNode(input, input2)
        with pytest.raises(Exception):
            self.compile(sum)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
    def test_buffer_tensor(self, dtype):
        def do_test(shape, dtype):
            node = hamkaas.BufferTensor(dtype, shape)
            model = self.compile(node)
            result = model.evaluate({})

            # Buffer contains garbage by default.
            assert list(result.shape) == shape
            assert result.dtype == dtype

        do_test([7], dtype)
        do_test([2, 2], dtype)
        do_test([10, 10, 10], dtype)
        do_test([100000], dtype)

        # 0-d tensors are not supported.
        with pytest.raises(Exception):
            do_test([], dtype)

        # 4-d tensors are not supported.
        with pytest.raises(Exception):
            do_test([2, 2, 2, 2], dtype)

        # int32 tensors are not supported.
        with pytest.raises(Exception):
            do_test([2, 2], torch.int32)

        # Empty tensors are not supported.
        with pytest.raises(Exception):
            do_test([0], dtype)

        # Something weird.
        with pytest.raises(Exception):
            do_test([2, 0, 2], dtype)
        with pytest.raises(Exception):
            do_test([1, -2, 3], dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_sum(self, dtype):
        def do_test(lhs_shape, rhs_shape, lhs_dtype, rhs_dtype=None):
            if rhs_dtype is None:
                rhs_dtype = dtype

            lhs = hamkaas.InputTensor("lhs", lhs_dtype, lhs_shape)
            rhs = hamkaas.InputTensor("rhs", rhs_dtype, rhs_shape)
            model = self.compile(lhs + rhs)

            lhs_tensor = torch.rand(lhs_shape, dtype=lhs_dtype)
            rhs_tensor = torch.rand(rhs_shape, dtype=rhs_dtype)
            result = model.evaluate({"lhs": lhs_tensor, "rhs": rhs_tensor})
            assert torch.allclose(result, lhs_tensor + rhs_tensor)

        # Simple addition.
        do_test([5], [5], dtype)
        do_test([2, 2], [2, 2], dtype)
        do_test([7, 7, 7], [7, 7, 7], dtype)

        # Broadcasting.
        do_test([5], [1], dtype)
        do_test([2, 2], [2, 1], dtype)
        do_test([2, 2], [1, 2], dtype)
        do_test([5, 5, 5], [1, 1, 1], dtype)
        do_test([5, 5, 5], [5, 1, 5], dtype)
        do_test([5, 5, 5], [1, 5, 1], dtype)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5], [5], torch.int64)

        # Incompatible types.
        with pytest.raises(Exception):
            do_test([5], [5], torch.float32, torch.float64)

        # Different number of dimensions.
        with pytest.raises(Exception):
            do_test([5], [5, 5], dtype)
        with pytest.raises(Exception):
            do_test([5], [5, 1], dtype)
        with pytest.raises(Exception):
            do_test([5], [1, 5], dtype)

        # Incompatible shapes.
        with pytest.raises(Exception):
            do_test([5], [4], dtype)
        with pytest.raises(Exception):
            do_test([2, 2], [2, 3], dtype)

        # Same number of elements, but different shapes.
        with pytest.raises(Exception):
            do_test([3, 2], [2, 3], dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_matmul(self, dtype):
        def do_test(lhs_shape, rhs_shape, lhs_dtype, rhs_dtype=None):
            if rhs_dtype is None:
                rhs_dtype = dtype

            lhs = hamkaas.InputTensor("lhs", lhs_dtype, lhs_shape)
            rhs = hamkaas.InputTensor("rhs", rhs_dtype, rhs_shape)
            model = self.compile(lhs @ rhs)

            lhs_tensor = torch.rand(lhs_shape, dtype=lhs_dtype)
            rhs_tensor = torch.rand(rhs_shape, dtype=rhs_dtype)
            result = model.evaluate({"lhs": lhs_tensor, "rhs": rhs_tensor})
            assert torch.allclose(result, lhs_tensor @ rhs_tensor)

        # Simple matrix multiplication.
        do_test([1, 1], [1, 1], dtype)
        do_test([2, 3], [3, 2], dtype)
        do_test([2, 5], [5, 7], dtype)

        # Matrix-vector product.
        do_test([1], [1, 1], dtype)
        do_test([3], [3, 2], dtype)
        do_test([10], [10, 7], dtype)

        # Batch matrix multiplication.
        do_test([1, 1, 1], [1, 1, 1], dtype)
        do_test([5, 5, 5], [5, 5, 5], dtype)
        do_test([2, 3, 5], [2, 5, 7], dtype)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([2, 2], [2, 2], torch.int64)

        # Incompatible types.
        with pytest.raises(Exception):
            do_test([2, 2], [2, 2], torch.float32, torch.float64)

        # Simple matrix multiplication incompatible shapes.
        with pytest.raises(Exception):
            do_test([2, 3], [2, 3], dtype)

        # Matrix-vector product incompatible shapes.
        with pytest.raises(Exception):
            do_test([2], [3, 2], dtype)

        # Batch matrix multiplication incompatible shapes.
        with pytest.raises(Exception):
            do_test([2, 3, 5], [2, 3, 5], dtype)
        with pytest.raises(Exception):
            do_test([2, 3, 3], [1, 3, 3], dtype)

        # Incompatible dimensions.
        with pytest.raises(Exception):
            do_test([2, 2], [2], dtype)
        with pytest.raises(Exception):
            do_test([2, 2, 2], [2], dtype)
        with pytest.raises(Exception):
            do_test([2, 2], [2, 2, 2], dtype)
        with pytest.raises(Exception):
            do_test([2], [2, 2, 2], dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_relu(self, dtype):
        def do_test(shape, dtype):
            input = hamkaas.InputTensor("input", dtype, shape)
            model = self.compile(input.relu())

            input_tensor = torch.rand(shape, dtype=dtype)
            result = model.evaluate({"input": input_tensor})
            assert torch.allclose(result, torch.relu(input_tensor))

        do_test([1], dtype)
        do_test([5], dtype)
        do_test([2, 2], dtype)
        do_test([5, 5, 5], dtype)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5], torch.int64)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_silu(self, dtype):
        def do_test(shape, dtype):
            input = hamkaas.InputTensor("input", dtype, shape)
            model = self.compile(input.silu())

            input_tensor = torch.rand(shape, dtype=dtype)
            result = model.evaluate({"input": input_tensor})
            silu = torch.nn.SiLU()
            assert torch.allclose(result, silu(input_tensor))

        do_test([1], dtype)
        do_test([5], dtype)
        do_test([2, 2], dtype)
        do_test([5, 5, 5], dtype)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5], torch.int64)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_slice_node(self, dtype):
        def do_test(shape, dtype, start, end):
            input = hamkaas.InputTensor("input", dtype, shape)
            model = self.compile(input[start:end])

            input_tensor = torch.rand(shape, dtype=dtype)
            result = model.evaluate({"input": input_tensor})
            assert torch.allclose(result, input_tensor[start:end])

        do_test([5], dtype, 1, 3)
        do_test([5], dtype, 0, 5)

        do_test([5, 7], dtype, 1, 3)
        do_test([5, 7], dtype, 0, 5)

        do_test([5, 7, 9], dtype, 1, 3)
        do_test([5, 7, 9], dtype, 0, 5)

        with pytest.raises(Exception):
            do_test([5], dtype, 1, 6)
        with pytest.raises(Exception):
            do_test([5], dtype, 2, 1)

        # Empty tensors are not supported.
        with pytest.raises(Exception):
            do_test([5], dtype, 0, 0)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5], torch.int64, 1, 3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_rms_norm(self, dtype):
        def do_test(lhs_shape, rhs_shape, lhs_dtype, rhs_dtype=None):
            if rhs_dtype is None:
                rhs_dtype = dtype

            input = hamkaas.InputTensor("input", lhs_dtype, lhs_shape)
            weights = hamkaas.InputTensor("weights", rhs_dtype, rhs_shape)
            model = self.compile(input.rms_norm(weights))

            input_tensor = torch.rand(lhs_shape, dtype=lhs_dtype)
            weights_tensor = torch.rand(rhs_shape, dtype=rhs_dtype)
            expected = torch.zeros_like(input_tensor)
            ss = 0.0
            for i in range(len(input_tensor)):
                ss += input_tensor[i] ** 2
            ss /= len(input_tensor)
            ss += 1e-5
            ss = 1.0 / math.sqrt(ss)
            for i in range(len(input_tensor)):
                expected[i] = weights_tensor[i] * (ss * input_tensor[i])

            actual = model.evaluate({"input": input_tensor, "weights": weights_tensor})
            assert torch.allclose(expected, actual)

        do_test([1], [1], dtype)
        do_test([5], [5], dtype)
        do_test([1000], [1000], dtype)

        # RMS norm is not supported for matrices.
        with pytest.raises(Exception):
            do_test([2, 2], [2, 2], dtype)

        # RMS norm requires weights of the same shape as the input.
        with pytest.raises(Exception):
            do_test([5], [4], dtype)

        # Incompatible types.
        with pytest.raises(Exception):
            do_test([5], [5], torch.float32, torch.float64)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5], [5], torch.int64)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_reshape(self, dtype):
        def do_test(shape, dtype, new_shape):
            input = hamkaas.InputTensor("input", dtype, shape)
            model = self.compile(input.reshape(new_shape))

            input_tensor = torch.rand(shape, dtype=dtype)
            result = model.evaluate({"input": input_tensor})
            assert torch.allclose(result, input_tensor.reshape(new_shape))

        do_test([5], dtype, [5])
        do_test([5], dtype, [1, 5])
        do_test([5], dtype, [5, 1])
        do_test([3, 4, 5], dtype, [6, 1, 10])
        do_test([6, 12], dtype, [3, 6, 4])

        # Incompatible shapes.
        with pytest.raises(Exception):
            do_test([5], dtype, [4])
        with pytest.raises(Exception):
            do_test([5], dtype, [2, 3])
        with pytest.raises(Exception):
            do_test([5], dtype, [5, 5])
        with pytest.raises(Exception):
            do_test([1], dtype, [0])
        with pytest.raises(Exception):
            do_test([1], dtype, [-1, -1])

        # Incompatible type.
        with pytest.raises(Exception):
            do_test([5], torch.int64, [5])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_complex_hadamard_product(self, dtype):
        def do_test(lhs_shape, rhs_shape, lhs_dtype, rhs_dtype=None):
            if rhs_dtype is None:
                rhs_dtype = lhs_dtype

            lhs = hamkaas.InputTensor("lhs", lhs_dtype, lhs_shape)
            rhs = hamkaas.InputTensor("rhs", rhs_dtype, rhs_shape)
            model = self.compile(lhs.complex_hadamard_product(rhs))

            lhs_tensor = torch.rand(lhs_shape, dtype=lhs_dtype)
            rhs_tensor = torch.rand(rhs_shape, dtype=rhs_dtype)
            expected = torch.zeros_like(lhs_tensor)
            for i in range(len(lhs_tensor)):
                expected[i][0] = lhs_tensor[i][0] * rhs_tensor[i][0] - lhs_tensor[i][1] * rhs_tensor[i][1]
                expected[i][1] = lhs_tensor[i][0] * rhs_tensor[i][1] + lhs_tensor[i][1] * rhs_tensor[i][0]

            actual = model.evaluate({"lhs": lhs_tensor, "rhs": rhs_tensor})
            assert torch.allclose(expected, actual)

        # Simple Hadamard product.
        do_test([5, 2], [5, 2], dtype)
        do_test([1, 2], [1, 2], dtype)

        # Wrong shapes.
        with pytest.raises(Exception):
            do_test([2], [2], dtype)
        with pytest.raises(Exception):
            do_test([2, 3], [2, 3], dtype)
        with pytest.raises(Exception):
            do_test([2, 2], [2, 3], dtype)
        with pytest.raises(Exception):
            do_test([2, 3], [2, 2], dtype)
        with pytest.raises(Exception):
            do_test([5, 5, 2], [5, 5, 2], dtype)

        # Incompatible types.
        with pytest.raises(Exception):
            do_test([5, 2], [5, 2], torch.float32, torch.float64)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5, 2], [5, 2], torch.int64)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_hadamard_product(self, dtype):
        def do_test(lhs_shape, rhs_shape, lhs_dtype, rhs_dtype=None):
            if rhs_dtype is None:
                rhs_dtype = lhs_dtype

            lhs = hamkaas.InputTensor("lhs", lhs_dtype, lhs_shape)
            rhs = hamkaas.InputTensor("rhs", rhs_dtype, rhs_shape)
            model = self.compile(lhs * rhs)

            lhs_tensor = torch.rand(lhs_shape, dtype=lhs_dtype)
            rhs_tensor = torch.rand(rhs_shape, dtype=rhs_dtype)
            expected = lhs_tensor * rhs_tensor

            actual = model.evaluate({"lhs": lhs_tensor, "rhs": rhs_tensor})
            assert torch.allclose(expected, actual)

        # Simple Hadamard product.
        do_test([1], [1], dtype)
        do_test([5], [5], dtype)
        do_test([5, 2], [5, 2], dtype)
        do_test([1, 2], [1, 2], dtype)
        do_test([1, 1, 1], [1, 1, 1], dtype)
        do_test([5, 7, 2], [5, 7, 2], dtype)

        # Incompatible shapes.
        with pytest.raises(Exception):
            do_test([2], [3], dtype)
        with pytest.raises(Exception):
            do_test([2, 3], [3, 2], dtype)
        with pytest.raises(Exception):
            do_test([2], [1, 2], dtype)
        with pytest.raises(Exception):
            do_test([2], [2, 1], dtype)

        # Incompatible types.
        with pytest.raises(Exception):
            do_test([5], [5], torch.float32, torch.float64)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5], [5], torch.int64)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_permute(self, dtype):
        def do_test(shape, dtype, permutation):
            input = hamkaas.InputTensor("input", dtype, shape)
            model = self.compile(input.permute(permutation))

            input_tensor = torch.rand(shape, dtype=dtype)
            result = model.evaluate({"input": input_tensor})
            assert torch.allclose(result, input_tensor.permute(permutation))

        do_test([5], dtype, [0])
        do_test([5, 7], dtype, [1, 0])
        do_test([5, 7, 9], dtype, [2, 0, 1])
        do_test([5, 7, 9], dtype, [0, 1, 2])
        do_test([5, 7, 9], dtype, [0, 2, 1])

        # Wrong permutation.
        with pytest.raises(Exception):
            do_test([5], dtype, [1])
        with pytest.raises(Exception):
            do_test([5], dtype, [0, 0])
        with pytest.raises(Exception):
            do_test([5], dtype, [0, 2])
        with pytest.raises(Exception):
            do_test([5, 7], dtype, [1, 1])
        with pytest.raises(Exception):
            do_test([5, 7, 9], dtype, [2, 0, 0])

        # Incompatible type.
        with pytest.raises(Exception):
            do_test([5], torch.int64, [0])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_replace_node(self, dtype):
        def do_test(shape, dtype, replacement_shape, replacement_dtype, start, end):
            input = hamkaas.InputTensor("input", dtype, shape)
            replacement = hamkaas.InputTensor("replacement", replacement_dtype, replacement_shape)
            model = self.compile(input.replace(replacement, start, end))

            input_tensor = torch.rand(shape, dtype=dtype)
            replacement_tensor = torch.rand(replacement_shape, dtype=replacement_dtype)
            expected = input_tensor.clone()
            expected[start:end] = replacement_tensor

            actual = model.evaluate({"input": input_tensor, "replacement": replacement_tensor})
            assert torch.allclose(expected, actual)

        do_test([5], dtype, [2], dtype, 1, 3)
        do_test([5], dtype, [5], dtype, 0, 5)

        # 2d tensor.
        with pytest.raises(Exception):
            do_test([5, 5], dtype, [2, 5], dtype, 1, 3)

        # Incompatible types.
        with pytest.raises(Exception):
            do_test([5], torch.float32, [2], torch.float64, 1, 3)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5], dtype, [2], torch.int64, 1, 3)

        # Wrong indices.
        with pytest.raises(Exception):
            do_test([5], dtype, [2], dtype, 1, 2)
        with pytest.raises(Exception):
            do_test([5], dtype, [2], dtype, 1, 6)
        with pytest.raises(Exception):
            do_test([5], dtype, [2], dtype, 4, 3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_buffer_node_persistence(self, dtype):
        buffer = hamkaas.BufferTensor(dtype, [5])
        replacement = hamkaas.InputTensor("replacement", dtype, [1])
        start_index = hamkaas.InputTensor("start_index", torch.int64, [1])
        end_index = hamkaas.InputTensor("end_index", torch.int64, [1])

        output = buffer.replace(replacement, start_index, end_index)
        model = self.compile(output)

        for i in range(5):
            result = model.evaluate({
                "replacement": torch.tensor([i], dtype=dtype),
                "start_index": torch.tensor([i], dtype=torch.int64),
                "end_index": torch.tensor([i + 1], dtype=torch.int64)
            })
            assert result[:i].allclose(torch.tensor(list(range(i)), dtype=dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_sliced_softmax(self, dtype):
        def do_test(shape, dtype, size):
            input = hamkaas.InputTensor("input", dtype, shape)
            model = self.compile(input.sliced_softmax(size))

            input_tensor = torch.rand(shape, dtype=dtype)
            softmax = torch.nn.Softmax(dim=0)
            expected = torch.cat((softmax(input_tensor[:size]), input_tensor[size:]))

            result = model.evaluate({"input": input_tensor})
            assert torch.allclose(result, expected)

        do_test([5], dtype, 3)
        do_test([5], dtype, 0)
        do_test([5], dtype, 5)

        # Incompatible shape.
        with pytest.raises(Exception):
            do_test([3, 3], dtype, 2)
        
        # Wrong index.
        with pytest.raises(Exception):
            do_test([5], dtype, 6)

        # Unsupported type.
        with pytest.raises(Exception):
            do_test([5], torch.int64, 3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("index", [1, 2, 5, 10, 20, 40, 60, 100])
    def test_fibonacci(self, dtype, index):
        x = hamkaas.ConstantTensor(torch.tensor([1], dtype=dtype))
        y = hamkaas.ConstantTensor(torch.tensor([1], dtype=dtype))

        x_t = torch.tensor([1], dtype=dtype)
        y_t = torch.tensor([1], dtype=dtype)

        for i in range(index):
            x, y = y, x + y
            x_t, y_t = y_t, x_t + y_t

        model = self.compile(y)
        result = model.evaluate({})

        assert torch.allclose(result, y_t)


class TestCompilerCuda(TestCompilerCpu):
    USE_GPU = True
