"""Kernel specification classes for TVM-FFI function parameters."""

from collections.abc import Sequence
from typing import Optional, Union

import tvm_ffi


class DefaultConfig:
    """Default configuration with context manager support."""

    _current: Optional["DefaultConfig"] = None
    _old_current: Optional["DefaultConfig"] = None
    device_type: str

    def __init__(self, *, device_type: Optional[str] = None) -> None:
        """Initialize a default configuration.

        Parameters
        ----------
        device_type : str, optional
            The device type (e.g., "cpu", "cuda", "metal"). If None, copies from current config.
        """
        if device_type is None:
            device_type = DefaultConfig.current().device_type  # type: ignore[union-attr]
        self.device_type = device_type

    def __enter__(self) -> "DefaultConfig":
        """Enter the context manager."""
        self._old_current = DefaultConfig._current
        DefaultConfig._current = self
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]
    ) -> None:
        """Exit the context manager."""
        DefaultConfig._current = self._old_current

    @classmethod
    def current(cls) -> Optional["DefaultConfig"]:
        """Get the current default configuration.

        Returns
        -------
        Optional[DefaultConfig]
            The current default configuration.
        """
        return cls._current

    @classmethod
    def _set_init_default_config(cls) -> None:
        """Set the initial default configuration."""
        current = cls.__new__(cls)
        current.device_type = "cuda"
        cls._current = current


# Initialize the default config to cuda
DefaultConfig._set_init_default_config()


class Param:
    """Base class for all parameters."""

    pass


class Var(Param):
    """variables: pointer, integer, floating-point, boolean, etc.

    Parameters
    ----------
    name : str
        The parameter name.
    dtype : str | tvm_ffi.dtype
        The data type of the parameter.

    """

    name: str
    dtype: tvm_ffi.dtype

    def __init__(self, name: str, dtype: Union[str, tvm_ffi.dtype]) -> None:
        """Initialize a Var parameter.

        Parameters
        ----------
        name : str
            The parameter name.
        dtype : str | tvm_ffi.dtype
            The data type of the parameter.

        """
        self.name = name
        self.dtype = tvm_ffi.dtype(dtype)


class Shape(Param):
    """Shape parameter.

    Parameters
    ----------
    name : str
        The parameter name.
    shape: list[int | Var]
        The shape of the parameter.
    dtype : str | tvm_ffi.dtype
        The data type of the parameter.

    """

    name: str
    shape: list[Union[int, Var]]

    def __init__(self, name: str, shape: list[Union[int, Var]]) -> None:
        """Initialize a Shape parameter.

        Parameters
        ----------
        name : str
            The parameter name.
        shape : list[int | Var]
            The shape of the parameter.
        dtype : str | tvm_ffi.dtype

        """
        self.name = name
        self.shape = shape


class Tensor(Param):
    """Tensor parameter.

    Parameters
    ----------
    name : str
        The parameter name.
    dtype : str | tvm_ffi.dtype
        The data type of the parameter.
    shape : Sequence[int | Var]
        The shape of the parameter.
    device_type : int
        The device type of the parameter.
    device_id : Var
        The device id of the parameter.
    strides : Optional[Sequence[Var]], optional
        The strides of the parameter, by default None.
    """

    name: str
    shape: list[Union[int, Var]]
    dtype: tvm_ffi.dtype
    strides: Optional[list[Var]]
    dlpack_device_type: int
    device_id: Var

    def __init__(
        self,
        name: str,
        shape: Sequence[Union[int, Var]],
        dtype: Union[str, tvm_ffi.dtype],
        *,
        device_type: Optional[str] = None,
        strides: Optional[Sequence[Var]] = None,
    ) -> None:
        """Initialize a Tensor parameter.

        Parameters
        ----------
        name : str
            The parameter name.
        device_type : str
            The device type of the parameter.
        shape : Sequence[int | Var]
            The shape of the parameter.
        dtype : str | tvm_ffi.dtype
            The data type of the parameter.
        strides : Optional[Sequence[Var]], optional
            The strides of the parameter, by default None.

        """
        self.name = name
        self.data = Var(name + ".data", tvm_ffi.dtype("handle"))
        self.shape: list[Union[int, Var]] = list(shape)
        self.dtype = tvm_ffi.dtype(dtype)
        self.strides: Optional[list[Var]] = list(strides) if strides is not None else None

        # Use default device type if none specified
        if device_type is None:
            device_type = DefaultConfig.current().device_type  # type: ignore[union-attr]

        example_device = tvm_ffi.device(device_type, 0)
        self.dlpack_device_type = example_device.dlpack_device_type()
        self.device_type_name = example_device.type
        self.device_id = Var(name + ".device_id", tvm_ffi.dtype("int32"))


class Stream(Param):
    """Stream parameter."""

    name: str
    var: Var

    def __init__(self, name: str) -> None:
        """Initialize a Stream parameter.

        Parameters
        ----------
        name : str
            The parameter name.
        """
        self.name = name
        self.var = Var(name, tvm_ffi.dtype("handle"))


def signature(name: str, params: list[Param]) -> str:
    """Generate a function signature string from name and parameters.

    Parameters
    ----------
    name : str
        The function name.
    params : list[Param]
        List of parameter objects (Var or Tensor).

    Returns
    -------
    str
        The formatted function signature.

    Raises
    ------
    ValueError
        If an unknown parameter type is encountered.

    """
    param_strs = []

    for param in params:
        if isinstance(param, Var):
            param_str = f"{param.name}: {param.dtype}"
        elif isinstance(param, Tensor):
            # Format tensor shape
            shape_strs = []
            for dim in param.shape:
                if isinstance(dim, Var):
                    shape_strs.append(dim.name)
                else:
                    shape_strs.append(str(dim))
            shape_str = "[" + ", ".join(shape_strs) + "]"
            param_str = f"{param.name}: Tensor({shape_str}, {param.dtype})"
        elif isinstance(param, Shape):
            # Format shape parameter
            shape_strs = []
            for dim in param.shape:
                if isinstance(dim, Var):
                    shape_strs.append(dim.name)
                else:
                    shape_strs.append(str(dim))
            shape_str = "[" + ", ".join(shape_strs) + "]"
            param_str = f"{param.name}: Shape({shape_str})"
        elif isinstance(param, Stream):
            param_str = f"{param.name}: Stream"
        else:
            raise RuntimeError(f"Unsupported parameter type: {type(param)}")

        param_strs.append(param_str)

    return f"{name}({', '.join(param_strs)})"
