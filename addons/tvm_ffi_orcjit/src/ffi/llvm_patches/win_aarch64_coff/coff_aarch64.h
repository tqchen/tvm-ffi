//===------ COFF_aarch64.h - JIT link functions for COFF/AArch64 ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_EXECUTIONENGINE_JITLINK_COFF_AARCH64_H
#define LLVM_EXECUTIONENGINE_JITLINK_COFF_AARCH64_H
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
namespace llvm::jitlink {
/// Build a graph from a COFF/AArch64 relocatable object. The buffer must
/// outlive the graph, which neither owns nor copies the underlying object
/// contents.
LLVM_ABI Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromCOFFObject_aarch64(
    MemoryBufferRef Buffer, std::shared_ptr<orc::SymbolStringPool> SSP);
LLVM_ABI void link_COFF_aarch64(std::unique_ptr<LinkGraph> G,
                                std::unique_ptr<JITLinkContext> Ctx);
} // namespace llvm::jitlink
#endif
