"""Call provider that implements a specific calling convention."""

from dataclasses import dataclass
from typing import Optional

from . import spec
from ._mlir import ir, llvm
from .mlir_builder import MLIRBuilder


@dataclass
class CallContext:
    """Call context that contains the information of the call."""

    # the function name
    fn_name: str
    # the module
    module: ir.Module
    # the function operation
    # can be used to insert llvm.alloca at beginning via
    # `with ir.InsertionPoint(entry_block.operations[0]):`
    entry_block: ir.Block
    # the parameters of the call
    params: list[spec.Param]
    # the current working stream through environment synced
    # with stream set in caller framework's stream context
    # this is queried by the tensor device type and id
    # useful for APIs where stream is not explicitly passed in
    env_stream: Optional[ir.Value]
    # the matched var binding
    matched_var_binding: dict[spec.Var, ir.Value]
    # raw arguments
    raw_args: list[ir.Value]
    # arg index
    raw_num_args: ir.Value
    # result
    raw_result: ir.Value


class CallProvider:
    """Call provider that implements a specific calling convention."""

    def __call__(self, current_block: ir.Block, context: CallContext) -> ir.Block:
        """Call the call provider.

        Parameters
        ----------
        current_block: ir.Block
            The current block to emit the call.

        context: CallContext
            The call context that contains the related information of the call.

        Returns
        -------
        ir.Block
            The new updated current block if any.
        """
        raise NotImplementedError("Call provider not implemented")


class NopCallProvider(CallProvider):
    """No-op call provider for testing purposes."""

    def __call__(self, current_block: ir.Block, context: CallContext) -> ir.Block:
        """No-op call provider that just returns the current block."""
        return current_block


class DynamicParamPackCallProvider(CallProvider, MLIRBuilder):
    """Packs dynamic arguments to a struct then calls the function.

    .. code-block:: c

        void call(Tensor0 t0, Tensor1 t1) {
            // packed arguments
            void** packed_args[] = {&t0, &t1};
            // call target
            target_func(packed_args);
        }

    Parameters
    ----------
    target_func: str
        The name of the target function.

    skip_first_stride: bool
        Whether to skip the first stride of the tensor.
        This is because the first stride is always 1, and it is not needed to pack.

    include_num_args: bool
        Whether to include the number of arguments in the packed arguments.

    struct_call: bool
        Whether to use the struct call convention.
    """

    def __init__(
        self, target_func: str, include_num_args: bool = False, struct_call: bool = False
    ) -> None:
        super().__init__()
        self.target_func = target_func
        self.include_num_args = include_num_args
        self.struct_call = struct_call

    def get_callee_struct_for_param_tensor(
        self,
        current_block: ir.Block,
        data: ir.Value,
        shape: list[ir.Value],
        strides: list[ir.Value],
        flatten_struct: ir.Type,
    ) -> ir.Type:
        """Routine used to override tensor passsing struct conention."""
        return flatten_struct

    def pack_param_tensor(
        self, current_block: ir.Block, context: CallContext, param: spec.Tensor
    ) -> tuple[ir.Type, ir.Value]:
        """Pack a tensor parameter to a struct."""
        data = context.matched_var_binding[param.data]
        shape = []
        strides = []
        # append all vars in shape
        for dim in param.shape:
            if isinstance(dim, spec.Var):
                shape.append(context.matched_var_binding[dim])
        # append all vars in strides
        if param.strides is not None:
            for dim in param.strides:
                if isinstance(dim, spec.Var):
                    strides.append(context.matched_var_binding[dim])
        flatten_struct, alloca = self.pack_values_to_alloca(
            current_block, context.entry_block, [data, *shape, *strides]
        )
        callee_struct = self.get_callee_struct_for_param_tensor(
            current_block, data, shape, strides, flatten_struct
        )

        return callee_struct, alloca

    def pack_param_var(
        self, current_block: ir.Block, context: CallContext, param: spec.Var
    ) -> tuple[ir.Type, ir.Value]:
        """Pack a var parameter to a struct."""
        value: ir.Value = context.matched_var_binding[param]
        _, alloca = self.pack_values_to_alloca(current_block, context.entry_block, [value])
        return (value.type, alloca)

    def pack_param_shape(
        self, current_block: ir.Block, context: CallContext, param: spec.Shape
    ) -> tuple[ir.Type, ir.Value]:
        """Pack a shape parameter to a struct."""
        dynamic_args: list[ir.Value] = []
        for dim in param.shape:
            if isinstance(dim, spec.Var):
                dynamic_args.append(context.matched_var_binding[dim])
        return self.pack_values_to_alloca(current_block, context.entry_block, dynamic_args)

    def pack_params(
        self, current_block: ir.Block, context: CallContext
    ) -> list[tuple[ir.Type, ir.Value]]:
        """Pack a parameter to a struct."""
        packed_params = []
        for index, param in enumerate(context.params):
            if isinstance(param, spec.Tensor):
                packed_params.append(self.pack_param_tensor(current_block, context, param))
            elif isinstance(param, spec.Var):
                packed_params.append(self.pack_param_var(current_block, context, param))
            elif isinstance(param, spec.Shape):
                packed_params.append(self.pack_param_shape(current_block, context, param))
            elif isinstance(param, spec.Stream):
                packed_params.append(self.pack_param_var(current_block, context, param.var))
            else:
                raise NotImplementedError(f"Unsupported parameter type: {type(param)}")
        return packed_params

    def __call__(self, current_block: ir.Block, context: CallContext) -> ir.Block:
        """Alloca call provider that uses dynamic param pack call convention."""
        packed_params = self.pack_params(current_block, context)

        if self.struct_call:
            # load back arguments as structs from alloca
            call_operands = []
            with ir.InsertionPoint(current_block):
                for struct_type, alloca in packed_params:
                    call_operands.append(llvm.load(struct_type, alloca))
        else:
            # pack the values to an alloca that we can pass as void**
            all_values = [value for _, value in packed_params]
            packed_args_type, packed_args_value = self.pack_values_to_alloca(
                current_block, context.entry_block, all_values
            )

            call_operands = [packed_args_value]
            if self.include_num_args:
                num_args = self.i32(len(all_values))
                call_operands.append(num_args)

        with ir.InsertionPoint(current_block):
            # Call the helper function
            llvm.call(
                result=None,
                callee=self.target_func,
                callee_operands=call_operands,
                op_bundle_sizes=[],
                op_bundle_operands=[],
            )

        return current_block
