I need to write a low-level FFI bridge that registers handles to JAX/XLA C API,
it needs to support

- variable length calls
- single handler whose behavior can be decided by the function name passed via ffi_call
- setup the function closure(let us say it is a custom struct i define) to context
  state in the intiialize stage
- during execute stage, retrieve the state(that contains the closure),
  go over the call frame arguments, cross check the argument type_index (e.g. if it is a Buffer),
  translate to DLTensor*  and then call the functor
- no need to mock up the headers since i will include them
-

Please provide an example that is self contained cc file, which implements this feature and register the handler.
note that i need to use the C API so i can access Call Frame and do switch on it.

Please setup the code using dummy examples AddOne(DLTensor* in, DLTensor* out)
and Mul(DLTensor* x, DLTensor* y, DLTensor* z)
