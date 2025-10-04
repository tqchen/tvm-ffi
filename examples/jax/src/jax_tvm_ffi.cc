#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Include the actual C API Headers from JAX/XLA and DLPack
// ============================================================================
#include <xla/ffi/api/c_api.h>
#include <tvm/ffi/function.h>
#include <dlpack/dlpack.h>

// ============================================================================
// C-style Implementations of Target Functions
// ============================================================================

// Forward declare the main execution handler.
XLA_FFI_Error* CustomCallExecute(XLA_FFI_CallFrame* frame);

void AddOne(DLTensor** args, int num_args) {
    printf("  -> Executing AddOne(in, out)\n");
    // In a real implementation, you would access args[0] (in) and args[1] (out)
    // and perform the computation.
}

void Mul(DLTensor** args, int num_args) {
    printf("  -> Executing Mul(x, y, z)\n");
    // Real implementation would use args[0], args[1], and args[2].
}

// ============================================================================
// FFI Bridge Implementation (Pure C)
// ============================================================================

// A C-style function pointer for our target functions.
typedef void (*FfiFunctor)(DLTensor**, int);

// A simple struct for our function registry lookup table.
typedef struct {
    const char* name;
    FfiFunctor func;
} FunctionRegistration;

// The "closure" or context state. This now stores the resolved functor.
typedef struct {
    FfiFunctor func;
} CustomCallState;

// A unique ID for our state struct, to be registered with the runtime.
static XLA_FFI_TypeId state_type_id = {0};

// A helper to create error objects using the C API.
static XLA_FFI_Error* CreateError(const XLA_FFI_Api* api, XLA_FFI_Error_Code code, const char* message) {
    XLA_FFI_Error_Create_Args err_args = {0};
    err_args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
    err_args.errc = code;
    err_args.message = message;
    return api->XLA_FFI_Error_Create(&err_args);
}

// The `instantiate` handler: called once per call site to resolve the function and set up state.
XLA_FFI_Error* CustomCallInstantiate(XLA_FFI_CallFrame* frame) {
    printf("[FFI Instantiate] Called with %lld attributes.\n", (long long)frame->attrs.size);
    // Register our state's type ID if it hasn't been already.
    if (state_type_id.type_id == 0) {
        XLA_FFI_TypeId_Register_Args args = {0};
        args.struct_size = XLA_FFI_TypeId_Register_Args_STRUCT_SIZE;
        args.name.ptr = "CustomCallState";
        args.name.len = strlen(args.name.ptr);
        args.type_id = &state_type_id;
        XLA_FFI_Error* err = frame->api->XLA_FFI_TypeId_Register(&args);
        if (err) return err;
    }

    // 1. Find the target function name from the attributes.
    const XLA_FFI_ByteSpan* name_span = NULL;
    for (int i = 0; i < frame->attrs.size; ++i) {
        const char* attr_name = "function_name";
        size_t attr_name_len = strlen(attr_name);
        if (frame->attrs.names[i]->len == attr_name_len &&
            strncmp(frame->attrs.names[i]->ptr, attr_name, attr_name_len) == 0) {

            if (frame->attrs.types[i] != XLA_FFI_AttrType_STRING) {
                return CreateError(frame->api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "'function_name' attribute must be a string");
            }
            name_span = (const XLA_FFI_ByteSpan*)frame->attrs.attrs[i];
            break;
        }
    }

    if (!name_span) {
        return CreateError(frame->api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "Missing 'function_name' attribute in instantiate stage");
    }

    // 2. Look up the function in a static registry.
    static const FunctionRegistration registry[] = {
        {"add_one", AddOne},
        {"mul", Mul}
    };
    static const int registry_size = sizeof(registry) / sizeof(registry[0]);

    FfiFunctor functor = NULL;
    for (int i = 0; i < registry_size; ++i) {
        const char* reg_name = registry[i].name;
        size_t reg_name_len = strlen(reg_name);
        if (reg_name_len == name_span->len &&
            strncmp(reg_name, name_span->ptr, name_span->len) == 0) {
            functor = registry[i].func;
            break;
        }
    }

    if (!functor) {
        char* target_name_buffer = (char*)malloc(name_span->len + 1);
        if (!target_name_buffer) return CreateError(frame->api, XLA_FFI_Error_Code_RESOURCE_EXHAUSTED, "Failed to allocate buffer for error message");
        memcpy(target_name_buffer, name_span->ptr, name_span->len);
        target_name_buffer[name_span->len] = '\0';

        char msg[256];
        snprintf(msg, sizeof(msg), "No custom call function registered for name '%s'", target_name_buffer);
        free(target_name_buffer);
        return CreateError(frame->api, XLA_FFI_Error_Code_NOT_FOUND, msg);
    }

    printf("[FFI Instantiate] Successfully resolved and stored functor for the call site.\n");

    // 3. Create the state struct on the heap and store the resolved functor.
    CustomCallState* state = (CustomCallState*)malloc(sizeof(CustomCallState));
    if (!state) {
        return CreateError(frame->api, XLA_FFI_Error_Code_RESOURCE_EXHAUSTED, "Failed to allocate state");
    }
    state->func = functor;

    // 4. Set the state in the execution context for the execute stage.
    XLA_FFI_State_Set_Args set_args = {0};
    set_args.struct_size = XLA_FFI_State_Set_Args_STRUCT_SIZE;
    set_args.ctx = frame->ctx;
    set_args.type_id = &state_type_id;
    set_args.state = state;
    set_args.deleter = free; // The runtime will call free(state) on teardown.

    return frame->api->XLA_FFI_State_Set(&set_args);
}

// The `execute` handler: called every time the operation runs.
XLA_FFI_Error* CustomCallExecute(XLA_FFI_CallFrame* frame) {
    printf("[FFI Execute] Called with %lld arguments.\n", (long long)frame->args.size);
    // 1. Retrieve the state that was created during the instantiate stage.
    CustomCallState* state;
    XLA_FFI_State_Get_Args get_args = {0};
    get_args.struct_size = XLA_FFI_State_Get_Args_STRUCT_SIZE;
    get_args.ctx = frame->ctx;
    get_args.type_id = &state_type_id;
    get_args.state = (void**)&state;
    XLA_FFI_Error* err = frame->api->XLA_FFI_State_Get(&get_args);
    if (err) return err;

    // 2. The functor is already resolved. Get it directly from the state.
    FfiFunctor functor = state->func;
    if (!functor) {
        // This should not happen if instantiate was successful.
        return CreateError(frame->api, XLA_FFI_Error_Code_INTERNAL, "Invalid state: functor is NULL");
    }
    printf("\n[FFI Execute] Calling pre-resolved functor.\n");

    // 3. Translate arguments from XLA_FFI_Buffer to DLTensor.
    printf("[FFI Execute] Processing %lld arguments.\n", (long long)frame->args.size);
    DLTensor* dl_tensors = (DLTensor*)malloc(frame->args.size * sizeof(DLTensor));
    DLTensor** dl_tensor_ptrs = (DLTensor**)malloc(frame->args.size * sizeof(DLTensor*));
    if (!dl_tensors || !dl_tensor_ptrs) {
        free(dl_tensors); free(dl_tensor_ptrs);
        return CreateError(frame->api, XLA_FFI_Error_Code_RESOURCE_EXHAUSTED, "Failed to allocate tensor arrays");
    }

    for (int i = 0; i < frame->args.size; ++i) {
        if (frame->args.types[i] != XLA_FFI_ArgType_BUFFER) {
            free(dl_tensors); free(dl_tensor_ptrs);
            return CreateError(frame->api, XLA_FFI_Error_Code_INVALID_ARGUMENT, "Argument is not a buffer");
        }
        XLA_FFI_Buffer* xla_buffer = (XLA_FFI_Buffer*)frame->args.args[i];
        dl_tensors[i].data = xla_buffer->data;
        dl_tensors[i].ndim = xla_buffer->rank;
        dl_tensors[i].shape = xla_buffer->dims;
        // Fill in dummy values
        // dl_tensors[i].device_type = 0; dl_tensors[i].device_id = 0;
        // dl_tensors[i].strides = NULL; dl_tensors[i].byte_offset = 0;
        // dl_tensors[i].dtype_code = 0; dl_tensors[i].dtype_bits = 32; dl_tensors[i].dtype_lanes = 1;

        dl_tensor_ptrs[i] = &dl_tensors[i];
    }

    // 4. Call the functor.
    functor(dl_tensor_ptrs, frame->args.size);

    free(dl_tensors);
    free(dl_tensor_ptrs);

    printf("[FFI Execute] Successfully executed call.\n");
    return NULL; // NULL signifies OK
}


void *CustomCallInstantiatePtr() {
  void *func = reinterpret_cast<void *>(CustomCallInstantiate);
  return func;
}

void *CustomCallExecutePtr() {
  void *func = reinterpret_cast<void *>(CustomCallExecute);
  return func;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(CustomCallInstantiatePtr, CustomCallInstantiatePtr);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(CustomCallExecutePtr, CustomCallExecutePtr);
