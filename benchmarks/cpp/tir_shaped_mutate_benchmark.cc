// A standalone mirror of TVM's TIR node hierarchy, so the StructuralMap engine can be measured
// on the real shape without depending on TVM.
//
// Mirrored exactly: ExprNode is a non-final base with a `ty` field, TreeNode kind and 64 child
// slots; VarNode derives from it and is FreeVar kind (so it takes the identity remap on every
// match); binary ops are final with two PrimExpr-typed operands; PrimExpr is a *view* over any
// ExprNode whose `ty` is a PrimType, with the same TypeTraits shape as tvm::TypedExpr.
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace tvm {
namespace ffi {

// ---------------------------------------------------------------- PrimType
class HPrimTypeObj : public Object {
 public:
  int64_t bits = 32;
  HPrimTypeObj() {}
  explicit HPrimTypeObj(int64_t bits) : bits(bits) {}
  explicit HPrimTypeObj(UnsafeInit) {}
  static void RegisterReflection() {
    reflection::ObjectDef<HPrimTypeObj>().def_ro("bits", &HPrimTypeObj::bits);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.PrimType", HPrimTypeObj, Object);
};
class HPrimType : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HPrimType, ObjectRef, HPrimTypeObj);
};

// ---------------------------------------------------------------- ExprNode
class HExprObj : public Object {
 public:
  Any ty;
  HExprObj() {}
  explicit HExprObj(Any ty) : ty(std::move(ty)) {}
  explicit HExprObj(UnsafeInit) {}
  static void RegisterReflection() {
    reflection::ObjectDef<HExprObj>().def_ro("ty", &HExprObj::ty);
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("h.Expr", HExprObj, Object);
};
class HExpr : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HExpr, ObjectRef, HExprObj);
};

// PrimExpr: a view over any HExprObj whose `ty` is an HPrimType. Mirrors tvm::TypedExpr.
class HPrimExpr : public HExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HPrimExpr, HExpr, HExprObj);
};

template <>
inline constexpr bool use_default_type_traits_v<HPrimExpr> = false;

template <>
struct TypeTraits<HPrimExpr> : public ObjectRefTypeTraitsBase<HPrimExpr> {
  using Base = ObjectRefTypeTraitsBase<HPrimExpr>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeStr;
  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return HPrimExpr::_type_is_nullable;
    if (src->type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
        !details::IsObjectInstance<HExprObj>(src->type_index)) {
      return false;
    }
    const auto* e = details::ObjectUnsafe::RawObjectPtrFromUnowned<HExprObj>(src->v_obj);
    return details::AnyUnsafe::CheckAnyStrict<HPrimType>(e->ty);
  }
  TVM_FFI_INLINE static std::optional<HPrimExpr> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }
};

// ---------------------------------------------------------------- Var (FreeVar)
class HVarObj : public HExprObj {
 public:
  int64_t id = 0;
  HVarObj(Any ty, int64_t id) : HExprObj(std::move(ty)), id(id) {}
  explicit HVarObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    // Mirrors tvm::MutateVar: consult the remap first, record on the way out.
    Expected<Any> cached = mutator->VarRemapGetExpected(value);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(cached);
    if (details::ExpectedUnsafe::GetData(cached).type_index() != TypeIndex::kTVMFFINone) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(cached));
    }
    Expected<void> set_result = mutator->VarRemapSetExpected(value, AnyView(value));
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(set_result).error()));
    }
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static TVMFFIAny StructuralVisit(StructuralVisitorObj*, AnyView) noexcept {
    // A Var is a leaf for the walk: nothing to descend into.
    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<Optional<VisitInterrupt>>(std::nullopt));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<HVarObj>().def_ro("id", &HVarObj::id);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<HVarObj>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&HVarObj::StructuralMutate)));
#ifdef BENCH_WALK
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::TypeAttrDef<HVarObj>().attr(
        refl::type_attr::kStructuralVisit,
        reinterpret_cast<void*>(static_cast<FStructuralVisit>(&HVarObj::StructuralVisit)));
#endif
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.Var";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HVarObj, HExprObj);
};
class HVar : public HPrimExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HVar, HPrimExpr, HVarObj);
};

// ---------------------------------------------------------------- IntImm
class HIntImmObj : public HExprObj {
 public:
  int64_t value_ = 0;
  HIntImmObj(Any ty, int64_t v) : HExprObj(std::move(ty)), value_(v) {}
  explicit HIntImmObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}
  static TVMFFIAny StructuralMutate(StructuralMutatorObj*, AnyView value) noexcept {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static TVMFFIAny StructuralVisit(StructuralVisitorObj*, AnyView) noexcept {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<Optional<VisitInterrupt>>(std::nullopt));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<HIntImmObj>().def_ro("value_", &HIntImmObj::value_);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<HIntImmObj>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&HIntImmObj::StructuralMutate)));
#ifdef BENCH_WALK
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::TypeAttrDef<HIntImmObj>().attr(
        refl::type_attr::kStructuralVisit,
        reinterpret_cast<void*>(static_cast<FStructuralVisit>(&HIntImmObj::StructuralVisit)));
#endif
  }
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.IntImm";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HIntImmObj, HExprObj);
};

// ------------------------------------------------------- binary ops (Add/Mul/FloorDiv/FloorMod)
template <typename T>
class HBinOpObj : public HExprObj {
 public:
  HPrimExpr a, b;
  HBinOpObj(Any ty, HPrimExpr a, HPrimExpr b)
      : HExprObj(std::move(ty)), a(std::move(a)), b(std::move(b)) {}
  explicit HBinOpObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}

  // Mirrors tvm::MutateBinary, including the guard that skips result-type inference when
  // neither operand's type changed.
  static Expected<HPrimType> ResultType(const HPrimExpr& a, const HPrimExpr& b) noexcept {
    auto at = a->ty.as<HPrimType>();
    auto bt = b->ty.as<HPrimType>();
    if (!at.has_value() || !bt.has_value() || at.value()->bits != bt.value()->bits) {
      return Unexpected(Error("TypeError", "mismatched types", ""));
    }
    return at.value();
  }
  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const T*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, a, mutator->MutateExpected(self->a));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimExpr, b, mutator->MutateExpected(self->b));
    if (a.same_as(self->a) && b.same_as(self->b)) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    }
    ObjectPtr<T> copy = make_object<T>(*static_cast<const T*>(self));
    if (!a->ty.same_as(self->a->ty) || !b->ty.same_as(self->b->ty)) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HPrimType, rty, ResultType(a, b));
      copy->HExprObj::ty = std::move(rty);
    }
    copy->a = std::move(a);
    copy->b = std::move(b);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }
  // Mirrors the mutate hook's shape on the visit side: descend both operands, propagate.
  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const auto* self = value.cast<const T*>();
    Expected<Optional<VisitInterrupt>> ra = visitor->VisitExpected(self->a);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(ra);
    Expected<Optional<VisitInterrupt>> rb = visitor->VisitExpected(self->b);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(rb);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(rb));
  }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<T>().attr(
        refl::type_attr::kStructuralMutate,
        reinterpret_cast<void*>(static_cast<FStructuralMutate>(&HBinOpObj<T>::StructuralMutate)));
#ifdef BENCH_WALK
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::TypeAttrDef<T>().attr(
        refl::type_attr::kStructuralVisit,
        reinterpret_cast<void*>(static_cast<FStructuralVisit>(&HBinOpObj<T>::StructuralVisit)));
#endif
  }
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
};
#define H_DECL_BINOP(Name, Key)                                      \
  class Name : public HBinOpObj<Name> {                              \
   public:                                                           \
    using HBinOpObj<Name>::HBinOpObj;                                \
    static constexpr const char* _type_key = Key;                    \
    TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(Name, HExprObj); \
  }
H_DECL_BINOP(HAddObj, "h.Add");
H_DECL_BINOP(HMulObj, "h.Mul");
H_DECL_BINOP(HFloorDivObj, "h.FloorDiv");
H_DECL_BINOP(HFloorModObj, "h.FloorMod");

TVM_FFI_STATIC_INIT_BLOCK() {
  HPrimTypeObj::RegisterReflection();
  HExprObj::RegisterReflection();
  HVarObj::RegisterReflection();
  HIntImmObj::RegisterReflection();
  HBinOpObj<HAddObj>::RegisterReflection();
  HBinOpObj<HMulObj>::RegisterReflection();
  HBinOpObj<HFloorDivObj>::RegisterReflection();
  HBinOpObj<HFloorModObj>::RegisterReflection();
}

// ------------------------------------------------------------------ lower bound
class HMinimalMutatorObj : public StructuralMutatorObj {
 public:
  HMinimalMutatorObj() : StructuralMutatorObj(VTable()) {}
  static TVMFFIAny Mutate(StructuralMutatorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn col(reflection::type_attr::kStructuralMutate);
    AnyView attr = col[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(self, value);
    }
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static TVMFFIAny NoRemapGet(StructuralMutatorObj*, AnyView) noexcept {
    TVMFFIAny a;
    a.type_index = TypeIndex::kTVMFFINone;
    a.zero_padding = 0;
    a.v_int64 = 0;
    return a;
  }
  static TVMFFIAny NoRemapSet(StructuralMutatorObj*, AnyView, AnyView) noexcept {
    TVMFFIAny a;
    a.type_index = TypeIndex::kTVMFFINone;
    a.zero_padding = 0;
    a.v_int64 = 0;
    return a;
  }
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable v{&HMinimalMutatorObj::Mutate, &HMinimalMutatorObj::Mutate,
                                           &HMinimalMutatorObj::NoRemapGet,
                                           &HMinimalMutatorObj::NoRemapSet};
    return &v;
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.MinimalMutator", HMinimalMutatorObj, StructuralMutatorObj);
};

// The walk-side counterpart of HMinimalMutatorObj: attr lookup then hook, nothing else. This is
// the lower bound a hooked StructuralWalk is measured against.
class HMinimalVisitorObj : public StructuralVisitorObj {
 public:
  HMinimalVisitorObj() : StructuralVisitorObj(VTable()) {}
  static TVMFFIAny Visit(StructuralVisitorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn col(reflection::type_attr::kStructuralVisit);
    AnyView attr = col[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralVisit>(attr.cast<void*>()))(self, value);
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<Optional<VisitInterrupt>>(std::nullopt));
  }
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable v{&HMinimalVisitorObj::Visit};
    return &v;
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.MinimalVisitor", HMinimalVisitorObj, StructuralVisitorObj);
};
}  // namespace ffi
}  // namespace tvm

using namespace tvm::ffi;
static Any g_ty;
static HVar g_outer{nullptr}, g_inner{nullptr}, g_repl{nullptr};
static HExpr g_root{nullptr};
size_t sink = 0;

template <typename T>
static HPrimExpr Bin(HPrimExpr a, HPrimExpr b) {
  return HPrimExpr(make_object<T>(g_ty, std::move(a), std::move(b)));
}
static HPrimExpr Imm(int64_t v) { return HPrimExpr(make_object<HIntImmObj>(g_ty, v)); }

// The same shape as TVM's split/fuse fixture: floordiv(o*16+i,32)*32 + floormod(o*16+i,32)
static void Build() {
  g_ty = Any(HPrimType(make_object<HPrimTypeObj>(32)));
  g_outer = HVar(make_object<HVarObj>(g_ty, 0));
  g_inner = HVar(make_object<HVarObj>(g_ty, 1));
  g_repl = HVar(make_object<HVarObj>(g_ty, 2));
  HPrimExpr q = Bin<HAddObj>(Bin<HMulObj>(g_outer, Imm(16)), g_inner);
  HPrimExpr r = Bin<HAddObj>(Bin<HMulObj>(g_outer, Imm(16)), g_inner);
  g_root = Bin<HAddObj>(Bin<HMulObj>(Bin<HFloorDivObj>(q, Imm(32)), Imm(32)),
                        Bin<HFloorModObj>(r, Imm(32)));
}
static size_t CountNodes(const HExpr& e) {
  const auto* o = e.as<HExprObj>();
  if (o == nullptr) return 0;
  if (const auto* add = e.as<HAddObj>()) return 1 + CountNodes(add->a) + CountNodes(add->b);
  if (const auto* mul = e.as<HMulObj>()) return 1 + CountNodes(mul->a) + CountNodes(mul->b);
  if (const auto* fd = e.as<HFloorDivObj>()) return 1 + CountNodes(fd->a) + CountNodes(fd->b);
  if (const auto* fm = e.as<HFloorModObj>()) return 1 + CountNodes(fm->a) + CountNodes(fm->b);
  return 1;
}
template <typename F>
static double BestNs(F&& f, int iters, int reps, int warm) {
  for (int i = 0; i < warm; ++i) f();
  std::vector<double> v;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = std::chrono::steady_clock::now();
    v.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / iters);
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
int main() {
  Build();
  const double n = static_cast<double>(CountNodes(g_root));
  auto replace_cb = [](const HVarObj* v, TVMFFIDefRegionKind) -> Expected<Any> {
    return v->id == 0 ? Any(g_repl) : Any(GetRef<HVar>(v));
  };
  Any chk = StructuralMap<WalkOrder::kPostOrder>(AnyView(g_root), replace_cb);
  printf("nodes=%.0f  replace produced a result=%d\n", n, (int)(chk.as<Object>() != nullptr));
  double mn = BestNs(
      [&] {
        ObjectPtr<HMinimalMutatorObj> m = make_object<HMinimalMutatorObj>();
        TVMFFIAny out = HMinimalMutatorObj::Mutate(m.get(), AnyView(g_root));
        Any owned = details::AnyUnsafe::MoveTVMFFIAnyToAny(&out);
        sink += owned.type_index();
      },
      2000, 15, 500);
  double rp = BestNs(
      [&] {
        Any v = StructuralMap<WalkOrder::kPostOrder>(AnyView(g_root), replace_cb);
        sink += v.type_index();
      },
      2000, 15, 500);
  // MUTATE "Never": a link keyed to a type that never occurs in the tree, so no node ever
  // matches. Isolates what the engine costs on unmatched nodes -- attr lookup, hook dispatch,
  // the per-node error check -- with no remap and no callback.
  auto never_mut_cb = [](const HPrimTypeObj*, TVMFFIDefRegionKind) -> Expected<Any> {
    return Any(nullptr);  // unreachable: no HPrimType node is in the walked tree
  };
  double nm = BestNs(
      [&] {
        Any v = StructuralMap<WalkOrder::kPostOrder>(AnyView(g_root), never_mut_cb);
        sink += v.type_index();
      },
      2000, 15, 500);

  // WALK arms, same three shapes.
#ifdef BENCH_WALK
  double wn = BestNs(
      [&] {
        ObjectPtr<HMinimalVisitorObj> v = make_object<HMinimalVisitorObj>();
        TVMFFIAny out = HMinimalVisitorObj::Visit(v.get(), AnyView(g_root));
        Any owned = details::AnyUnsafe::MoveTVMFFIAnyToAny(&out);
        sink += owned.type_index();
      },
      2000, 15, 500);
  auto never_walk_cb = [](const HPrimTypeObj*) -> Expected<WalkResult> {
    return WalkResult::Advance();
  };
  double wnever = BestNs(
      [&] {
        auto r = StructuralWalkExpected<WalkOrder::kPostOrder>(AnyView(g_root), never_walk_cb);
        sink += r.has_value() ? 1 : 0;
      },
      2000, 15, 500);
  size_t seen = 0;
  auto match_walk_cb = [&seen](const HVarObj*) -> Expected<WalkResult> {
    ++seen;
    return WalkResult::Advance();
  };
  double wmatch = BestNs(
      [&] {
        auto r = StructuralWalkExpected<WalkOrder::kPostOrder>(AnyView(g_root), match_walk_cb);
        sink += r.has_value() ? 1 : 0;
      },
      2000, 15, 500);

#endif
  printf("\n--- MUTATE ---------------------------------------------\n");
  printf("minimal-vtable floor        : %8.2f ns/node\n", mn / n);
  printf("hooked Never   (no match)   : %8.2f ns/node   over floor %+7.2f\n", nm / n,
         (nm - mn) / n);
  printf("hooked Replace-Var          : %8.2f ns/node   over floor %+7.2f\n", rp / n,
         (rp - mn) / n);
  printf("  matched-path cost (R-N)   : %+8.2f ns/node\n", (rp - nm) / n);
#ifdef BENCH_WALK
  printf("\n--- WALK -----------------------------------------------\n");
  printf("minimal-visitor floor       : %8.2f ns/node\n", wn / n);
  printf("hooked Never   (no match)   : %8.2f ns/node   over floor %+7.2f\n", wnever / n,
         (wnever - wn) / n);
  printf("hooked Match-Var            : %8.2f ns/node   over floor %+7.2f\n", wmatch / n,
         (wmatch - wn) / n);
  printf("  matched-path cost (M-N)   : %+8.2f ns/node   (vars seen=%zu)\n", (wmatch - wnever) / n,
         seen);
#endif
  return 0;
}
