// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TVM_FFI_ORCJIT_COFF_LLVM_ABI_H_
#define TVM_FFI_ORCJIT_COFF_LLVM_ABI_H_
#include "llvm/Config/llvm-config.h"
#include "llvm/Config/abi-breaking.h"
#if LLVM_VERSION_MAJOR != 23 || LLVM_VERSION_MINOR != 1 || LLVM_VERSION_PATCH != 1
#error "The COFF backport requires the LLVM 23.1.1 private class definitions"
#endif
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#error "The COFF backport requires the release static SDK ABI"
#endif
#if defined(_MSC_VER)
#include <yvals_core.h>
#if _ITERATOR_DEBUG_LEVEL != 0 || !defined(_DLL) || defined(_DEBUG)
#error "The COFF backport requires release /MD and _ITERATOR_DEBUG_LEVEL=0"
#endif
#if defined(_CPPRTTI)
#error "The COFF backport must match the static SDK's disabled RTTI"
#endif
#endif
#endif
