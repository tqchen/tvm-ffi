//===------ COFF_aarch64.cpp - JIT linker for COFF/AArch64
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// LLVM issue: not yet filed. Affected version: LLVM 23.1.1.
// Trigger: Windows ARM64 COFF objects loaded through ORC ObjectLinkingLayer.
// Symptom: LLVM rejects ARM64 COFF graph construction and linker dispatch.
// ## Removal
// Delete this directory and the Windows ARM64 source block in the addon CMake
// file once the minimum LLVM SDK provides native AArch64 COFF JITLink support.
// See README.md for the pinned private ABI and static archive boundary.

#include "coff_aarch64.h"
#include "abi.h"
#include "upstream/COFFLinkGraphBuilder.h"
#include "upstream/JITLinkGeneric.h"
#include "upstream/SEHFrameSupport.h"
#include "llvm/ExecutionEngine/JITLink/COFF.h"
#include "llvm/ExecutionEngine/JITLink/aarch64.h"
#include "llvm/Support/Endian.h"

#define DEBUG_TYPE "jitlink"

using namespace llvm;
using namespace llvm::jitlink;

namespace {
// LLVM 23.1.1 has no public platform-edge sentinel. Keep its public enum intact.
constexpr Edge::Kind FirstCOFFRelocation =
    aarch64::RequestTLSDescEntryAndTransformToPageOffset12 + 1;
static_assert(FirstCOFFRelocation > aarch64::PageOffset12);
enum EdgeKind_coff_aarch64 : Edge::Kind {
  Pointer32NB = FirstCOFFRelocation,
  SectionIdx16,
  SecRel32,
  SecRelLow12A,
  SecRelHigh12A,
  SecRelLow12L,
};

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case Pointer32NB:
    return "Pointer32NB";
  case SectionIdx16:
    return "SectionIdx16";
  case SecRel32:
    return "SecRel32";
  case SecRelLow12A:
    return "SecRelLow12A";
  case SecRelHigh12A:
    return "SecRelHigh12A";
  case SecRelLow12L:
    return "SecRelLow12L";
  default:
    return aarch64::getEdgeKindName(K);
  }
}

class COFFLinkGraphBuilder_aarch64 : public COFFLinkGraphBuilder {
  Error addRelocations() override {
    for (const auto &S : sections())
      if (auto Err = forEachRelocation(
              S, this, &COFFLinkGraphBuilder_aarch64::addRelocation))
        return Err;
    return Error::success();
  }

  Error addRelocation(const object::RelocationRef &R,
                      const object::SectionRef &S, Block &B) {
    using namespace support::endian;
    if (R.getType() == COFF::IMAGE_REL_ARM64_ABSOLUTE)
      return Error::success();
    auto SI = R.getSymbol();
    if (SI == getObject().symbol_end())
      return make_error<JITLinkError>(
          "Invalid symbol in AArch64 COFF relocation");
    auto CS = getObject().getCOFFSymbol(*SI);
    auto *Target = getGraphSymbol(getObject().getSymbolIndex(CS));
    if (!Target)
      return make_error<JITLinkError>("Missing AArch64 COFF relocation target");
    uint64_t Offset =
        S.getAddress() + R.getOffset() - B.getAddress().getValue();
    unsigned Width = R.getType() == COFF::IMAGE_REL_ARM64_ADDR64    ? 8
                     : R.getType() == COFF::IMAGE_REL_ARM64_SECTION ? 2
                                                                    : 4;
    if (Offset > B.getSize() || B.getSize() - Offset < Width)
      return make_error<JITLinkError>(
          "AArch64 COFF relocation exceeds its block");
    if (B.isZeroFill())
      return make_error<JITLinkError>(
          "AArch64 COFF relocation in zero-fill block");
    const char *P = B.getContent().data() + Offset;
    uint32_t Instr = Width == 4 ? read32le(P) : 0;
    uint32_t ClearMask = 0;
    Edge::Kind K = Edge::Invalid;
    int64_t A = 0;
    switch (R.getType()) {
    case COFF::IMAGE_REL_ARM64_ADDR64:
      K = aarch64::Pointer64;
      A = read64le(P);
      break;
    case COFF::IMAGE_REL_ARM64_ADDR32:
      K = aarch64::Pointer32;
      A = int32_t(read32le(P));
      break;
    case COFF::IMAGE_REL_ARM64_ADDR32NB:
      K = Pointer32NB;
      A = int32_t(read32le(P));
      if (!GetImageBaseSymbol()(getGraph()))
        addImageBaseSymbol();
      break;
    case COFF::IMAGE_REL_ARM64_REL32:
      K = aarch64::Delta32;
      A = int64_t(int32_t(read32le(P))) - 4;
      break;
    case COFF::IMAGE_REL_ARM64_REL21:
    case COFF::IMAGE_REL_ARM64_PAGEBASE_REL21:
      K = R.getType() == COFF::IMAGE_REL_ARM64_REL21 ? aarch64::ADRLiteral21
                                                     : aarch64::Page21;
      if ((Instr & 0x9f000000) !=
          (K == aarch64::Page21 ? 0x90000000 : 0x10000000))
        return make_error<JITLinkError>(
            "Invalid ADR/ADRP relocation instruction");
      // COFF stores a signed byte addend, including in an ADRP instruction.
      // Page rounding applies to S+A, not separately to S and A.
      A = SignExtend64<21>(((Instr >> 29) & 3) | ((Instr >> 3) & 0x1ffffc));
      ClearMask = 0x60ffffe0;
      break;
    case COFF::IMAGE_REL_ARM64_PAGEOFFSET_12A:
    case COFF::IMAGE_REL_ARM64_SECREL_LOW12A:
    case COFF::IMAGE_REL_ARM64_SECREL_HIGH12A:
      if ((Instr & 0x7f000000) != 0x11000000)
        return make_error<JITLinkError>("Invalid ADD relocation instruction");
      K = R.getType() == COFF::IMAGE_REL_ARM64_PAGEOFFSET_12A
              ? Edge::Kind(aarch64::PageOffset12)
          : R.getType() == COFF::IMAGE_REL_ARM64_SECREL_LOW12A ? SecRelLow12A
                                                               : SecRelHigh12A;
      A = (Instr >> 10) & 0xfff;
      ClearMask = 0x003ffc00;
      break;
    case COFF::IMAGE_REL_ARM64_PAGEOFFSET_12L:
    case COFF::IMAGE_REL_ARM64_SECREL_LOW12L:
      if (!aarch64::isLoadStoreImm12(Instr))
        return make_error<JITLinkError>(
            "Invalid load/store relocation instruction");
      K = R.getType() == COFF::IMAGE_REL_ARM64_PAGEOFFSET_12L
              ? Edge::Kind(aarch64::PageOffset12)
              : SecRelLow12L;
      A = ((Instr >> 10) & 0xfff) << aarch64::getPageOffset12Shift(Instr);
      ClearMask = 0x003ffc00;
      break;
    case COFF::IMAGE_REL_ARM64_BRANCH26:
      if ((Instr & 0x7c000000) != 0x14000000)
        return make_error<JITLinkError>("Invalid B/BL relocation instruction");
      K = aarch64::Branch26PCRel;
      A = SignExtend64<28>((Instr & 0x03ffffff) << 2);
      ClearMask = 0x03ffffff;
      break;
    case COFF::IMAGE_REL_ARM64_BRANCH19:
      if (!aarch64::isCondBranchImm19(Instr) &&
          !aarch64::isCompAndBranchImm19(Instr))
        return make_error<JITLinkError>(
            "Invalid conditional branch relocation instruction");
      K = aarch64::CondBranch19PCRel;
      A = SignExtend64<21>(((Instr >> 5) & 0x7ffff) << 2);
      ClearMask = 0x00ffffe0;
      break;
    case COFF::IMAGE_REL_ARM64_BRANCH14:
      if (!aarch64::isTestAndBranchImm14(Instr))
        return make_error<JITLinkError>(
            "Invalid test branch relocation instruction");
      K = aarch64::TestAndBranch14PCRel;
      A = SignExtend64<16>(((Instr >> 5) & 0x3fff) << 2);
      ClearMask = 0x0007ffe0;
      break;
    case COFF::IMAGE_REL_ARM64_SECTION:
      K = SectionIdx16;
      A = int16_t(read16le(P));
      if (CS.isUndefined())
        return make_error<JITLinkError>(
            "SECTION relocation requires a defined target");
      Target = &getGraph().addAbsoluteSymbol(
          "secidx",
          orc::ExecutorAddr(CS.isAbsolute()
                                ? getObject().getNumberOfSections() + 1
                                : CS.getSectionNumber()),
          2, Linkage::Strong, Scope::Local, false);
      break;
    case COFF::IMAGE_REL_ARM64_SECREL:
      K = SecRel32;
      A = int32_t(read32le(P));
      break;
    default:
      return make_error<JITLinkError>("Unsupported AArch64 COFF relocation: " +
                                      Twine(R.getType()));
    }
    if (K >= SecRel32 && !Target->isDefined())
      return make_error<JITLinkError>(
          "Section-relative relocation requires a defined target");
    if (ClearMask) {
      if (Offset % 4)
        return make_error<JITLinkError>(
            "Unaligned AArch64 instruction relocation");
      write32le(B.getMutableContent(getGraph()).data() + Offset,
                Instr & ~ClearMask);
    }
    B.addEdge(K, Offset, *Target, A);
    return Error::success();
  }

public:
  COFFLinkGraphBuilder_aarch64(const object::COFFObjectFile &Obj,
                               std::shared_ptr<orc::SymbolStringPool> SSP,
                               Triple TT, SubtargetFeatures Features)
      : COFFLinkGraphBuilder(Obj, std::move(SSP), std::move(TT),
                             std::move(Features), getEdgeKindName) {}
};

// Named COFF imports are local pointer slots, so imported data and functions
// both retain their real address and ADRP accesses stay near their use sites.
Error buildTables(LinkGraph &G) {
  SmallVector<Symbol *, 8> Imports;
  for (auto *S : G.external_symbols())
    if ((*S->getName()).starts_with("__imp_"))
      Imports.push_back(S);
  if (!Imports.empty()) {
    auto &IAT = G.createSection("$__IAT", orc::MemProt::Read);
    for (auto *S : Imports) {
      auto Name = G.intern((*S->getName()).drop_front(6));
      auto *Target = G.findExternalSymbolByName(Name);
      if (!Target)
        Target = G.findDefinedSymbolByName(Name);
      if (!Target)
        Target = G.findAbsoluteSymbolByName(Name);
      if (!Target)
        Target = &G.addExternalSymbol(Name, 0, false);
      auto &Slot = aarch64::createAnonymousPointer(G, IAT, Target);
      G.makeDefined(*S, Slot.getBlock(), 0, 8, Linkage::Strong, Scope::Local,
                    true);
    }
  }
  SmallVector<Edge *, 16> Branches;
  for (auto *B : G.blocks())
    for (auto &E : B->edges())
      if (E.getKind() == aarch64::Branch26PCRel && !E.getTarget().isDefined())
        Branches.push_back(&E);
  if (!Branches.empty()) {
    auto &Pointers = G.createSection("$__GOT", orc::MemProt::Read);
    auto &Stubs =
        G.createSection("$__STUBS", orc::MemProt::Read | orc::MemProt::Exec);
    DenseMap<std::pair<Symbol *, int64_t>, Symbol *> Entries;
    for (auto *E : Branches) {
      auto [I, Inserted] =
          Entries.try_emplace({&E->getTarget(), E->getAddend()}, nullptr);
      if (Inserted) {
        auto &Pointer = aarch64::createAnonymousPointer(
            G, Pointers, &E->getTarget(), E->getAddend());
        I->second = &aarch64::createAnonymousPointerJumpStub(G, Stubs, Pointer);
      }
      E->setTarget(*I->second);
      E->setAddend(0);
    }
  }
  return Error::success();
}

class COFFJITLinker_aarch64 : public JITLinker<COFFJITLinker_aarch64> {
  friend class JITLinker<COFFJITLinker_aarch64>;

public:
  using JITLinker::JITLinker;

private:
  Error applyFixup(LinkGraph &G, Block &B, const Edge &E) const {
    using namespace support::endian;
    if (E.getKind() < FirstCOFFRelocation)
      return aarch64::applyFixup(G, B, E, nullptr);
    uint64_t Value = E.getTarget().getAddress().getValue() + E.getAddend();
    char *P = B.getAlreadyMutableContent().data() + E.getOffset();
    if (E.getKind() == SectionIdx16) {
      if (!isUInt<16>(Value))
        return makeTargetOutOfRangeError(G, B, E);
      write16le(P, Value);
      return Error::success();
    }
    if (E.getKind() == Pointer32NB) {
      auto *Base = GetImageBaseSymbol()(G);
      if (!Base)
        return make_error<JITLinkError>("Missing __ImageBase symbol");
      Value -= Base->getAddress().getValue();
    } else {
      if (!E.getTarget().isDefined())
        return make_error<JITLinkError>(
            "Section-relative relocation requires a defined target");
      Value -= SectionRange(E.getTarget().getSection()).getStart().getValue();
    }
    if (E.getKind() == Pointer32NB || E.getKind() == SecRel32) {
      if (!isUInt<32>(Value))
        return makeTargetOutOfRangeError(G, B, E);
      write32le(P, Value);
    } else {
      uint32_t Instr = read32le(P);
      unsigned Shift = E.getKind() == SecRelLow12L
                           ? aarch64::getPageOffset12Shift(Instr)
                           : 0;
      if (E.getKind() == SecRelHigh12A) {
        Value >>= 12;
        if (!isUInt<12>(Value))
          return makeTargetOutOfRangeError(G, B, E);
      } else {
        Value &= 0xfff;
        if (Value & ((1U << Shift) - 1))
          return make_error<JITLinkError>(
              "Unaligned section-relative load/store offset");
        Value >>= Shift;
      }
      write32le(P, Instr | (Value << 10));
    }
    return Error::success();
  }
};
} // namespace

namespace llvm::jitlink {
Expected<std::unique_ptr<LinkGraph>> createLinkGraphFromCOFFObject_aarch64(
    MemoryBufferRef Buffer, std::shared_ptr<orc::SymbolStringPool> SSP) {
  auto Obj = object::ObjectFile::createCOFFObjectFile(Buffer);
  if (!Obj)
    return Obj.takeError();
  auto Features = (*Obj)->getFeatures();
  if (!Features)
    return Features.takeError();
  return COFFLinkGraphBuilder_aarch64(
             **Obj, std::move(SSP), (*Obj)->makeTriple(), std::move(*Features))
      .buildGraph();
}

void link_COFF_aarch64(std::unique_ptr<LinkGraph> G,
                       std::unique_ptr<JITLinkContext> Ctx) {
  PassConfiguration Config;
  if (Ctx->shouldAddDefaultTargetPasses(G->getTargetTriple())) {
    if (auto MarkLive = Ctx->getMarkLivePass(G->getTargetTriple())) {
      Config.PrePrunePasses.push_back(std::move(MarkLive));
      Config.PrePrunePasses.push_back(SEHFrameKeepAlivePass(".pdata"));
    } else {
      Config.PrePrunePasses.push_back(markAllSymbolsLive);
    }
    Config.PostPrunePasses.push_back(buildTables);
  }
  if (auto Err = Ctx->modifyPassConfig(*G, Config))
    return Ctx->notifyFailed(std::move(Err));
  COFFJITLinker_aarch64::link(std::move(Ctx), std::move(G), std::move(Config));
}
} // namespace llvm::jitlink
