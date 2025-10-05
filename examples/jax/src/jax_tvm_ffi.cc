#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <xla/ffi/api/ffi.h>

#include <array>
#include <memory>
#include <type_traits>

// subclass of xla::ffi::Ffi
// so we can reuse existing helper functions defined.
// we do not use the Handler class directly since we need to operate
// on lower-level call frame arguments.
class JAXTVMFFIHandler : public xla::ffi::Ffi {
 public:
  explicit JAXTVMFFIHandler(tvm::ffi::Function func, int traits) : func_(func), traits_(traits) {}

  XLA_FFI_Error* Call(XLA_FFI_CallFrame* call_frame) const final {
    // If passed a call frame with the metadata extension, just return the
    if (XLA_FFI_PREDICT_FALSE(call_frame->extension_start != nullptr &&
                              call_frame->extension_start->type == XLA_FFI_Extension_Metadata)) {
      return PopulateMetadata(call_frame->api, reinterpret_cast<XLA_FFI_Metadata_Extension*>(
                                                   call_frame->extension_start));
    }
    std::cout << "FFIHandler::Call called " << std::endl;

    return Success();
  }

 private:
  // populate metadata extension to the call frame
  XLA_FFI_Error* PopulateMetadata(const XLA_FFI_Api* api,
                                  XLA_FFI_Metadata_Extension* extension) const {
    if (XLA_FFI_Error* err = StructSizeIsGreaterOrEqual(api, "XLA_FFI_Metadata_Extension",
                                                        XLA_FFI_Metadata_Extension_STRUCT_SIZE,
                                                        extension->extension_base.struct_size)) {
      return err;
    }
    std::cout << "FFIHandler::PopulateMetadata called " << std::endl;

    if (XLA_FFI_Error* err =
            StructSizeIsGreaterOrEqual(api, "XLA_FFI_Metadata", XLA_FFI_Metadata_STRUCT_SIZE,
                                       extension->metadata->struct_size)) {
      return err;
    }

    extension->metadata->api_version = XLA_FFI_Api_Version{
        XLA_FFI_Api_Version_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        XLA_FFI_API_MAJOR,
        XLA_FFI_API_MINOR,
    };
    // TODO: add traits that indicate command buffer compatible
    // which may be needed for cudagraph compatibility
    extension->metadata->traits = static_cast<XLA_FFI_Handler_Traits>(traits_);
    return Success();
  }

  static XLA_FFI_Error* Success() { return nullptr; }
  // underlying function
  tvm::ffi::Function func_;
  int traits_;
};

//-------------------------------------------------------------------
// Macro to generate trampoline table
// We use trampoline table to work around the limitation that xla
// ffi handler right now can only take in raw function pointer
// without extra closure data during registration.
//
// We will generate 1000 function pointers that calls into
// JAXTVMFFIRegistry::Global()->Call(index, call_frame);
// and then we can dispatch based on the index.
//
// Likely it should be sufficient since very unlikely users
// will have more than 1000 handlers.
//-------------------------------------------------------------------
// global registry of handlers
class JAXTVMFFIRegistry {
 public:
  static void* Register(tvm::ffi::Function func, int traits) {
    return Global()->RegisterInternal(func, traits);
  }

  static size_t RegisteredCount() {
    return Global()->registered_count_;
  }

 private:
  // size of the trampoline table
  static constexpr int kTrampolineTableSize = 1024;
  // current number of handlers allocated
  size_t registered_count_ = 0;
  // handler table to dispatch to
  std::array<std::unique_ptr<JAXTVMFFIHandler>, kTrampolineTableSize> handler_table_;
  // global static trampoline table pre-populated
  std::array<XLA_FFI_Handler*, kTrampolineTableSize> trampoline_table_ =
      MakeTrampolineTable(std::make_index_sequence<kTrampolineTableSize>{});

  // global instance
  static JAXTVMFFIRegistry* Global() {
    static JAXTVMFFIRegistry* inst = new JAXTVMFFIRegistry();
    return inst;
  }

  // internal register function
  void* RegisterInternal(tvm::ffi::Function func, int traits) {
    if (registered_count_ >= kTrampolineTableSize) {
      TVM_FFI_THROW(RuntimeError) << "JAXTVMFFIRegistry: cannot register more than "
                                  << kTrampolineTableSize << " handlers";
    }
    handler_table_[registered_count_++] = std::make_unique<JAXTVMFFIHandler>(func, traits);
    return reinterpret_cast<void*>(trampoline_table_[registered_count_ - 1]);
  }

  // must not inline the entry to minimize the trampoline function size
  TVM_FFI_NO_INLINE static XLA_FFI_Error* Entry(int index, XLA_FFI_CallFrame* call_frame) {
    return Global()->handler_table_[index]->Call(call_frame);
  }
  // trampoline functions that dispatches based on index
  // We use this design to work around the limitation that xla ffi handler right now
  // can only take in raw function pointer without extra user data
  template <int index>
  static XLA_FFI_Error* Trampoline(XLA_FFI_CallFrame* call_frame) {
    return JAXTVMFFIRegistry::Entry(index, call_frame);
  }
  // helper to geenrate the trampoline table
  template <size_t... Indices>
  static std::array<XLA_FFI_Handler*, sizeof...(Indices)> MakeTrampolineTable(
      std::index_sequence<Indices...>) {
    // This uses a parameter pack expansion to create an array of function pointers,
    // one for each index in the sequence.
    return std::array<XLA_FFI_Handler*, sizeof...(Indices)>{&Trampoline<Indices>...};
  }
};

// callback to register a handler to the global registry
TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_tvm_ffi_handler, JAXTVMFFIRegistry::Register);
// get the number of handlers registered
TVM_FFI_DLL_EXPORT_TYPED_FUNC(registered_count, JAXTVMFFIRegistry::RegisteredCount);
