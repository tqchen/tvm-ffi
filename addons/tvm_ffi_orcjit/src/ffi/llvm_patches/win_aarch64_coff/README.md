# AArch64 COFF JITLink backport

LLVM issue: not yet filed. Affected release: LLVM 23.1.1.

LLVM 23.1.1 rejects AArch64 COFF both when building a LinkGraph and when
linking it. ObjectLinkingLayer has no public backend registration hook.
This Windows ARM64-only backport supplies those two dispatch cases and an
AArch64 COFF backend without modifying the installed LLVM SDK.

`coff_dispatch.cc` replaces the complete static `COFF.cpp` archive member:
`createLinkGraphFromCOFFObject`, `link_COFF`, and
`GetImageBaseSymbol::operator()`. CMake adds the replacement as direct addon
objects, before archive extraction. The original member must remain unselected;
whole-archive loading and duplicate-symbol suppression are not supported.
The x86-64 dispatch still calls LLVM's existing backend. Addon DLL exports
remain restricted to the existing explicit TVM-FFI API.

The `upstream` headers are unchanged private declarations from
[llvmorg-23.1.1](https://github.com/llvm/llvm-project/tree/llvmorg-23.1.1/llvm/lib/ExecutionEngine/JITLink).
`COFFOptions.inc` is generated from that release's `COFFOptions.td` using
`llvm-tblgen -gen-opt-parser-defs`. Their class layouts must match the stock
library: exact LLVM version, static archive, release MSVC iterator mode,
release DLL CRT, disabled RTTI and ABI-breaking checks. Build guards reject
incompatible modes. Public SDK headers are used unchanged; local platform
edge kinds follow the pinned last AArch64 edge. No SDK archive is rewritten.

The backend implements ARM64 COFF address, page, branch and section-relative
relocations. Named imports use graph-local pointer slots for both data and
functions. External B/BL edges use local thunks, with any addend absorbed into
the thunk's pointer. Conditional branches, graph-defined branches and direct
ADR/ADRP references retain their architectural displacement limits.

Normal ObjectLinkingLayer ownership and cleanup still apply. Existing
init/fini and unwind-stripping patches remain responsible for the consumer's
CRT behavior. This backport does not register SEH tables or enable Windows
exception unwinding.

## Removal

Once the minimum LLVM SDK has native AArch64 COFF JITLink support, delete this
directory and its Windows ARM64 CMake source/guard/license block. Remove the
matching source-distribution include and narrow lint exceptions. Keep host
import address resolution until the embedding can use COFFPlatform normally.

The LLVM-derived sources retain their Apache-2.0 WITH LLVM-exception license;
see `LICENSE.txt`. The copied private headers are intentionally not reformatted.
