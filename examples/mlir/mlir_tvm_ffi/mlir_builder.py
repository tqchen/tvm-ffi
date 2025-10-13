"""MLIR type builder and basic operations for LLVM dialect."""

from collections.abc import Sequence
from typing import Any, Optional

from ._mlir import ir, llvm


class MLIRTypeBuilder:
    """Builder for MLIR types and basic operations."""

    def __init__(self) -> None:
        """Initialize the MLIR type builder."""
        self.i32_type = ir.IntegerType.get_signless(32)
        self.ui32_type = ir.IntegerType.get_unsigned(32)
        self.i64_type = ir.IntegerType.get_signless(64)
        self.i16_type = ir.IntegerType.get_signless(16)
        self.i8_type = ir.IntegerType.get_signless(8)
        self.i1_type = ir.IntegerType.get_signless(1)
        self.f32_type = ir.Type.parse("f32")
        self.f64_type = ir.Type.parse("f64")
        self.ptr_type = llvm.PointerType.get()
        self.gpu_ptr_type = llvm.PointerType.get(address_space=1)
        # did not find a programmatic way to get the void type
        self.void_type = ir.Type.parse("!llvm.void")

    def as_attr(self, tp: ir.Type) -> ir.TypeAttr:
        """Convert the type to a type attribute."""
        return ir.TypeAttr.get(tp)

    def int_type(self, bits: int) -> ir.Type:
        """Get the `i<bits>` type."""
        return ir.IntegerType.get_signless(bits)

    def uint_type(self, bits: int) -> ir.Type:
        """Get the `ui<bits>` type."""
        return ir.IntegerType.get_unsigned(bits)

    def struct_type(
        self,
        *,
        name: Optional[str] = None,
        fields: Sequence[ir.Type] = (),
        packed: bool = False,
    ) -> ir.Type:
        """Get or create a struct type.

        Parameters
        ----------
        name : Optional[str]
            The name of the struct type. If not provided, a `literal` struct type is created,
            which is identified by its fields only.
        fields : Sequence[ir.Type]
            The fields of the struct type.
        packed : bool
            Whether to create a packed struct type. If `True`, there is no padding between fields.
            Otherwise, there might be padding
            between fields to ensure alignment.

        See Also
        --------
        https://mlir.llvm.org/docs/Dialects/LLVM/#structure-types

        """
        if name is None:
            return llvm.StructType.get_literal(fields, packed=packed)
        else:
            return llvm.StructType.new_identified(name, fields, packed=packed)

    def identified_struct_type(self, name: str) -> ir.Type:
        """Get a previously created identified struct type by its name."""
        return llvm.StructType.get_identified(name)

    def func_type(self, *, params: Sequence[ir.Type] = (), ret: ir.Type) -> ir.Type:
        """Get a function type.

        Parameters
        ----------
        params : Sequence[ir.Type]
            The parameters of the function type.
        ret : ir.Type
            The return type of the function type.

        See Also
        --------
        https://mlir.llvm.org/docs/Dialects/LLVM/#function-types

        """
        return ir.Type.parse(
            "!llvm.func<{} ({})>".format(str(ret), ", ".join(map(str, params)))
        )  # did not find a programmatic way to get the function type


class MLIRBuilder(MLIRTypeBuilder):
    """A builder for MLIR related types and operations.

    Convention:
    all statement-generation methods expect we are inside a insertion point of the current_block.
    If the statement terminates the current block, it will assign the new block to the current_block
    but not set the insersion point.
    """

    def __init__(self) -> None:
        """Initialize the MLIR builder."""
        super().__init__()
        self.module: Optional[ir.Module] = None
        self.const_str_table: dict[str, ir.Value] = {}
        self.get_element_extra_kwargs: dict[str, Any] = {}

    # create constants
    def integer_constant(self, tp: ir.Type, value: int) -> ir.Value:
        """Create an integer constant with the given type and value."""
        return llvm.ConstantOp(tp, ir.IntegerAttr.get(tp, value)).res

    def i32(self, value: int) -> ir.Value:
        """Create an i32 constant with the given value."""
        return self.integer_constant(self.i32_type, value)

    def ui32(self, value: int) -> ir.Value:
        """Create a ui32 constant with the given value."""
        return self.integer_constant(self.ui32_type, value)

    def i1(self, value: int) -> ir.Value:
        """Create an i1 constant with the given value."""
        return self.integer_constant(self.i1_type, value)

    def i8(self, value: int) -> ir.Value:
        """Create an i8 constant with the given value."""
        return self.integer_constant(self.i8_type, value)

    def i16(self, value: int) -> ir.Value:
        """Create an i16 constant with the given value."""
        return self.integer_constant(self.i16_type, value)

    def i64(self, value: int) -> ir.Value:
        """Create an i64 constant with the given value."""
        return self.integer_constant(self.i64_type, value)

    def mul(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Create a multiplication operation between two values."""
        return llvm.mul(lhs, rhs, overflow_flags=llvm.IntegerOverflowFlags.none)

    # expressions
    def not_equal(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Create a not-equal comparison between two values."""
        return llvm.icmp(llvm.ICmpPredicate.ne, lhs, rhs)

    def equal(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Create an equal comparison between two values."""
        return llvm.icmp(llvm.ICmpPredicate.eq, lhs, rhs)

    def or_(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Create a logical OR operation between two values."""
        return llvm.or_(lhs, rhs)

    def and_(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        """Create a logical AND operation between two values."""
        return llvm.and_(lhs, rhs)

    def not_(self, value: ir.Value) -> ir.Value:
        """Create a logical NOT operation."""
        # Ensure we're working with i1 type for boolean operations
        if value.type != self.i1_type:
            value = llvm.trunc(res=self.i1_type, arg=value)
        return llvm.xor(value, self.i1(1))

    def br(self, target_block: ir.Block, *, args: Optional[list[ir.Value]] = None) -> None:
        """Create an unconditional branch.

        Parameters
        ----------
        target_block : ir.Block
            The target block to branch to.
        args : list[ir.Value], optional
            The values to pass as arguments to the target block. If None,
            no arguments are passed. The target block must have the same
            number of arguments as values in this list.

        """
        if args is None:
            llvm.br(dest=target_block)
        else:
            llvm.br(dest_operands=args, dest=target_block)

    def address_of(self, name: str, tp: ir.Type) -> ir.Value:
        """Get the address of a global symbol."""
        return llvm.AddressOfOp(tp, name).result

    def getelementptr(
        self, ptr: ir.Value, constant_indices: Sequence[int], elem_type: ir.Type
    ) -> ir.Value:
        """Create a getelementptr operation.

        Parameters
        ----------
        ptr : ir.Value
            The pointer to the element.
        indices : Sequence[ir.Value]
            The indices to the element.
        elem_type : ir.Type
            The type of the element.

        Returns
        -------
        ir.Value
            The resulting element pointer.
        """
        try:
            return llvm.getelementptr(
                self.ptr_type,
                ptr,
                [],
                raw_constant_indices=ir.DenseI32ArrayAttr.get(constant_indices),
                elem_type=elem_type,
                **self.get_element_extra_kwargs,
            )
        except Exception:
            # compatibility with different LLVM versions
            self.get_element_extra_kwargs = {"no_wrap_flags": []}
            return llvm.getelementptr(
                self.ptr_type,
                ptr,
                [],
                raw_constant_indices=ir.DenseI32ArrayAttr.get(constant_indices),
                elem_type=elem_type,
                **self.get_element_extra_kwargs,
            )

    def return_(self, ret: Optional[ir.Value] = None) -> None:
        """Create a return statement."""
        llvm.return_(arg=ret)

    def cond_br(self, cond: ir.Value, true_block: ir.Block, false_block: ir.Block) -> None:
        """Create a conditional branch."""
        llvm.cond_br(
            cond,
            true_dest_operands=[],
            false_dest_operands=[],
            true_dest=true_block,
            false_dest=false_block,
        )

    def define_global_string(self, content: str) -> str:
        """Define a global string symbol with the given content using standard MLIR APIs."""
        if content in self.const_str_table:
            return self.const_str_table[content]
        symbol = f"__tvm_ffi__str_{len(self.const_str_table)}"

        # The string_attr.value gives us the original string, but we need to escape it for
        # MLIR parsing. Let's use a simple approach: escape quotes and backslashes
        escaped_content = content.replace("\\", "\\\\").replace('"', '\\"')

        # Parse the MLIR string with proper escaping using the standard API
        module_body = self.module.body  # type: ignore[union-attr]
        with ir.InsertionPoint(module_body):
            parsed_op = ir.Operation.parse(
                f'llvm.mlir.global private constant @{symbol}("{escaped_content}\\00")'
            )
            module_body.append(parsed_op)
            self.const_str_table[content] = symbol
        return symbol

    # function
    def function(
        self, name: str, params_type: Sequence[ir.Type], ret_type: ir.Type, internal: bool = False
    ) -> tuple[list[ir.Value], ir.Block]:
        """Create a function with the given signature."""
        func_op = llvm.func(
            name,
            function_type=self.as_attr(self.func_type(ret=ret_type, params=params_type)),
        )
        if internal:
            func_op.attributes["llvm.linkage"] = ir.StringAttr.get("private")
        else:
            func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        params = []
        func_body: Any = func_op.body
        if func_body is not None:
            entry_block = ir.Block.create_at_start(func_body)
        else:
            raise RuntimeError("Function body is None")
        for param_type in params_type:
            params.append(entry_block.add_argument(param_type, ir.Location.unknown()))
        return params, entry_block

    def declare_extern_func(self, name: str, params: Sequence[ir.Type], ret: ir.Type) -> None:
        """Define extern function."""
        func_type = self.func_type(params=params, ret=ret)
        func_op = llvm.func(
            name,
            function_type=self.as_attr(func_type),
        )
        func_op.attributes["llvm.linkage"] = ir.StringAttr.get("external")

    def pack_values_to_alloca(
        self,
        current_block: ir.Block,
        entry_block: ir.Block,
        values: Sequence[ir.Value],
    ) -> ir.Value:
        """Pack values to an alloca that lays out in the order of the values.

        Parameters
        ----------
        current_block : ir.Block
            The current block.
        entry_block : ir.Block
            The entry block to create an alloca for the struct.
        values : Sequence[ir.Value]
            The values to pack.

        Returns
        -------
        tuple[ir.Type, ir.Value]
            The struct type and the alloca.
        """
        with ir.InsertionPoint(entry_block.operations[0]):
            # declare the struct type
            struct_type = self.struct_type(fields=[value.type for value in values])
            alloca = llvm.alloca(
                res=self.ptr_type,
                elem_type=struct_type,
                array_size=self.i32(1),
            )

        with ir.InsertionPoint(current_block):
            for index, value in enumerate(values):
                # Get pointer to the field at the given index
                field_ptr = self.getelementptr(
                    alloca,
                    [0, index],
                    struct_type,
                )
                # Store the value into the field
                llvm.store(value, field_ptr)
        return (struct_type, alloca)

    def find_func_in_module(self, module: ir.Module, name: str) -> Optional[ir.Operation]:
        """Find a function in the module."""
        for op in module.body:  # type: ignore[union-attr]
            if op.name == "llvm.func":
                # Get the function name from the sym_name attribute
                if "sym_name" in op.attributes:
                    func_name = str(op.attributes["sym_name"]).strip('"')
                    if func_name == name:
                        return op
        return None
