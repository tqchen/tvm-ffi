"""Test cases for the spec module."""

from mlir_tvm_ffi import spec  # type: ignore[import-not-found]


def test_signature() -> None:
    """Test signature generation with symbolic names in tensor dimensions."""
    n_var = spec.Var("n", "int32")
    params = [spec.Tensor("x", [1, 2, n_var], "float32"), spec.Var("y", "int32"), spec.Stream("s")]
    result = spec.signature("func", params)
    expected = "func(x: Tensor([1, 2, n], float32), y: int32, s: Stream)"
    assert result == expected


def test_default_config() -> None:
    """Test the DefaultConfig class functionality."""
    # Test with CPU default using context manager
    with spec.DefaultConfig(device_type="cpu"):
        cpu_tensor = spec.Tensor("cpu_tensor", [2, 4], "float32")
        assert cpu_tensor.device_type_name == "cpu"
        assert spec.DefaultConfig.current().device_type == "cpu"

    # Test with CUDA default (should be default)
    cuda_tensor = spec.Tensor("cuda_tensor", [2, 4], "float32")
    assert cuda_tensor.device_type_name == "cuda"
    assert spec.DefaultConfig.current().device_type == "cuda"

    # Test explicit device type overrides default
    with spec.DefaultConfig(device_type="cpu"):
        explicit_cuda_tensor = spec.Tensor("explicit_cuda", [2, 4], "float32", device_type="cuda")
        assert explicit_cuda_tensor.device_type_name == "cuda"
        # Default should still be CPU
        assert spec.DefaultConfig.current().device_type == "cpu"

    # Test that DefaultConfig() copies from current
    with spec.DefaultConfig(device_type="cpu"):
        config = spec.DefaultConfig()
        assert config.device_type == "cpu"

    # Test context manager state restoration
    with spec.DefaultConfig(device_type="cpu"):
        cpu_tensor2 = spec.Tensor("cpu_tensor2", [2, 4], "float32")
        assert cpu_tensor2.device_type_name == "cpu"

    # Should be back to cuda default
    cuda_tensor2 = spec.Tensor("cuda_tensor2", [2, 4], "float32")
    assert cuda_tensor2.device_type_name == "cuda"
