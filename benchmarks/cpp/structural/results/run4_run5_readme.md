# Rounds 4 and 5: impact of always-inlining the ObjectPtr destructor chain

Both runs were made by the task owner session directly (see #387 Progress Log), on the
round-3 tree `c9bb47d` with header-only source edits, hooks and engines untouched.

- `compare_full_run4_reset_inline.md`: real-TVM harness, six states, 21 processes each, all
  eleven fixtures. `old`/`gold`/`uc` are the round-3 binaries unchanged; `*-ri` are the same
  sources plus apache/tvm-ffi#764 (`TVM_FFI_INLINE` on `ObjectPtr::reset` and
  `WeakObjectPtr::reset`), with `libtvm_ffi`/`tvm_compiler` rebuilt in a second build
  directory with the round-3 options.
- `compare_mini_run5_macro_inline.md`: mini-TIR harness (`split_fuse_bench`), twelve states,
  21 processes each: base, `-ri` (#764), `-rim` (#764 plus `TVM_FFI_INLINE` on the four
  members of `TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN`), `-rimd` (`-rim` plus an
  explicit `TVM_FFI_INLINE ~TypeName() = default;` in that macro). Executables compiled
  against the `-ri` libraries; nothing on the mini hot path lives in the libraries.

Static inventory of `BinaryMutate<AddNode>` per state is in the #387 Progress Log.
