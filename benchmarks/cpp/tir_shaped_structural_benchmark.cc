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

// Minimal TIR-shaped harness for structural traversal cost.
//
// Mirrors just enough of TVM's expression hierarchy to make the engine behave as it does on real
// TIR: a non-final Expr base carrying a `ty` field, a FreeVar-kind Var that takes an identity
// remap on every match, and binary nodes that rebuild copy-on-write. Nothing else.
//
// Fixture is the expression a fuse followed by a split produces:
//
//     floordiv(q, 32) * 32 + floormod(r, 32),  q = outer * 16 + inner
//
//   shared   : r is the same object as q      -> a DAG
//   distinct : r is a structurally equal copy -> a tree
//
// Both are 17 traversal visits. They differ only in how many land on the same object, which is
// what separates a deduplicating traversal from one that does not.
//
// Each path is measured as
//
//     ns_per_visit = median_traversal_time / measured_visit_count
//
// where the visit count is INSTRUMENTED per path, never assumed: the old visitor dedups, the
// structural traversals do not, and dividing both by one node count compares different workloads.

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace ffi {

// --------------------------------------------------------------------------- nodes
class HPrimTypeObj : public Object {
 public:
  int64_t bits = 32;
  explicit HPrimTypeObj(int64_t bits) : bits(bits) {}
  explicit HPrimTypeObj(UnsafeInit) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.PrimType", HPrimTypeObj, Object);
};
class HPrimType : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HPrimType, ObjectRef, HPrimTypeObj);
};

// Non-final base with child slots and a `ty` field, as ExprNode has.
class HExprObj : public Object {
 public:
  Any ty;
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

// FreeVar kind: the engine takes an identity remap on every match, as it does for tir::Var.
class HVarObj : public HExprObj {
 public:
  int64_t id = 0;
  HVarObj(Any ty, int64_t id) : HExprObj(std::move(ty)), id(id) {}
  explicit HVarObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}
  static TVMFFIAny VarVisit(StructuralVisitorObj*, AnyView) noexcept {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<Optional<VisitInterrupt>>(std::nullopt));
  }
  static TVMFFIAny VarMutate(StructuralMutatorObj*, AnyView value) noexcept {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<HVarObj>().def_ro("id", &HVarObj::id);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::TypeAttrDef<HVarObj>().attr(refl::type_attr::kStructuralVisit,
                                      reinterpret_cast<void*>(&HVarObj::VarVisit));
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<HVarObj>().attr(refl::type_attr::kStructuralMutate,
                                      reinterpret_cast<void*>(&HVarObj::VarMutate));
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.Var";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HVarObj, HExprObj);
};
class HVar : public HExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HVar, HExpr, HVarObj);
};

class HIntImmObj : public HExprObj {
 public:
  int64_t value = 0;
  HIntImmObj(Any ty, int64_t v) : HExprObj(std::move(ty)), value(v) {}
  explicit HIntImmObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}
  static TVMFFIAny IntImmVisit(StructuralVisitorObj*, AnyView) noexcept {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<Optional<VisitInterrupt>>(std::nullopt));
  }
  static TVMFFIAny IntImmMutate(StructuralMutatorObj*, AnyView value) noexcept {
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<HIntImmObj>().def_ro("value", &HIntImmObj::value);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::TypeAttrDef<HIntImmObj>().attr(refl::type_attr::kStructuralVisit,
                                         reinterpret_cast<void*>(&HIntImmObj::IntImmVisit));
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<HIntImmObj>().attr(refl::type_attr::kStructuralMutate,
                                         reinterpret_cast<void*>(&HIntImmObj::IntImmMutate));
  }
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.IntImm";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HIntImmObj, HExprObj);
};

// One template covers Add/Mul/FloorDiv/FloorMod; they differ only in type key.
template <typename T>
class HBinOpObj : public HExprObj {
 public:
  HExpr a, b;
  HBinOpObj(Any ty, HExpr a, HExpr b) : HExprObj(std::move(ty)), a(std::move(a)), b(std::move(b)) {}
  explicit HBinOpObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}
  static TVMFFIAny BinVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const auto* self = value.cast<const T*>();
    Expected<Optional<VisitInterrupt>> ra = visitor->VisitExpected(self->a);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(ra);
    Expected<Optional<VisitInterrupt>> rb = visitor->VisitExpected(self->b);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(rb);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(rb));
  }
  // Copy-on-write rebuild. A map preserves node types, so `ty` carries over from the copy.
  static TVMFFIAny BinMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const T*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HExpr, a, mutator->MutateExpected(self->a));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(HExpr, b, mutator->MutateExpected(self->b));
    if (a.same_as(self->a) && b.same_as(self->b)) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    }
    ObjectPtr<T> copy = make_object<T>(*static_cast<const T*>(self));
    copy->a = std::move(a);
    copy->b = std::move(b);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }
  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::TypeAttrDef<T>().attr(refl::type_attr::kStructuralVisit,
                                reinterpret_cast<void*>(&HBinOpObj<T>::BinVisit));
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<T>().attr(refl::type_attr::kStructuralMutate,
                                reinterpret_cast<void*>(&HBinOpObj<T>::BinMutate));
  }
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
};
#define H_BINOP(Name, Key)                                           \
  class Name : public HBinOpObj<Name> {                              \
   public:                                                           \
    using HBinOpObj<Name>::HBinOpObj;                                \
    static constexpr const char* _type_key = Key;                    \
    TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(Name, HExprObj); \
  }
H_BINOP(HAddObj, "h.Add");
H_BINOP(HMulObj, "h.Mul");
H_BINOP(HFloorDivObj, "h.FloorDiv");
H_BINOP(HFloorModObj, "h.FloorMod");
#undef H_BINOP

// A type the fixture never contains, for the never-matching lambda.
class HNeverObj : public HExprObj {
 public:
  explicit HNeverObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.Never";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HNeverObj, HExprObj);
};
class HNever : public HExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HNever, HExpr, HNeverObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  HExprObj::RegisterReflection();
  HVarObj::RegisterReflection();
  HIntImmObj::RegisterReflection();
  HBinOpObj<HAddObj>::RegisterReflection();
  HBinOpObj<HMulObj>::RegisterReflection();
  HBinOpObj<HFloorDivObj>::RegisterReflection();
  HBinOpObj<HFloorModObj>::RegisterReflection();
}

}  // namespace ffi
}  // namespace tvm

using namespace tvm::ffi;

// --------------------------------------------------------------------------- Call
// Mirrors tir::CallNode: operands live in an Array, so mutating one goes through the engine's
// seq-container path, which allocates a fresh SeqObj and back-fills the unchanged prefix as soon
// as any element differs. Fixed-arity nodes never touch that path.
namespace tvm {
namespace ffi {
class HCallObj : public HExprObj {
 public:
  Array<Any> args;
  HCallObj(Any ty, Array<Any> args) : HExprObj(std::move(ty)), args(std::move(args)) {}
  explicit HCallObj(UnsafeInit) : HExprObj(UnsafeInit{}) {}
  static TVMFFIAny CallVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const auto* self = value.cast<const HCallObj*>();
    Expected<Optional<VisitInterrupt>> r = visitor->VisitExpected(self->args);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(r);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(r));
  }
  static TVMFFIAny CallMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const HCallObj*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Array<Any>, args, mutator->MutateExpected(self->args));
    if (args.same_as(self->args)) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    }
    ObjectPtr<HCallObj> copy = make_object<HCallObj>(*self);
    copy->args = std::move(args);
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
  }
  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<HCallObj>().def_ro("args", &HCallObj::args);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::TypeAttrDef<HCallObj>().attr(refl::type_attr::kStructuralVisit,
                                       reinterpret_cast<void*>(&HCallObj::CallVisit));
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::TypeAttrDef<HCallObj>().attr(refl::type_attr::kStructuralMutate,
                                       reinterpret_cast<void*>(&HCallObj::CallMutate));
  }
  static constexpr uint32_t _type_child_slots = 0;
  static constexpr bool _type_final = true;
  static constexpr const char* _type_key = "h.Call";
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(HCallObj, HExprObj);
};

// --------------------------------------------------------------------------- floor
// The lower bound a hooked traversal is measured against: attr-column lookup, one indirect call
// into the registered hook, nothing else. It is not "no work" -- the hook bodies still run.
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
  static TVMFFIAny NoRemap(StructuralMutatorObj*, AnyView) noexcept {
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static TVMFFIAny NoRemapSet(StructuralMutatorObj*, AnyView, AnyView) noexcept {
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable v{&Mutate, &Mutate, &NoRemap, &NoRemapSet};
    return &v;
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.MinimalMutator", HMinimalMutatorObj, StructuralMutatorObj);
};
class HMinimalVisitorObj : public StructuralVisitorObj {
 public:
  HMinimalVisitorObj() : StructuralVisitorObj(VTable()) {}
  static TVMFFIAny Visit(StructuralVisitorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn col(reflection::type_attr::kStructuralVisit);
    AnyView attr = col[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralVisit>(attr.cast<void*>()))(self, value);
    }
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable v{&Visit};
    return &v;
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("h.MinimalVisitor", HMinimalVisitorObj, StructuralVisitorObj);
};

TVM_FFI_STATIC_INIT_BLOCK() { HCallObj::RegisterReflection(); }

}  // namespace ffi
}  // namespace tvm

// --------------------------------------------------------------------------- builders
namespace {
template <typename T>
HExpr Bin(const Any& ty, HExpr a, HExpr b) {
  return HExpr(make_object<T>(ty, std::move(a), std::move(b)));
}
HExpr Call2(const Any& ty, HExpr a, HExpr b) {
  return HExpr(make_object<HCallObj>(ty, Array<Any>{Any(std::move(a)), Any(std::move(b))}));
}
}  // namespace

// --------------------------------------------------------------------------- old path
// A deduplicating visitor and a non-memoizing mutator, matching PostOrderVisit and
// StmtExprMutator. Both dispatch on type index, as ExprFunctor's vtable does.
namespace {

class OldVisitor {
 public:
  explicit OldVisitor(std::function<void(const HExpr&)> cb) : cb_(std::move(cb)) {}
  void Visit(const HExpr& e) {
    if (!visited_.insert(e.get()).second) return;  // dedup, as PostOrderVisit does
    const int32_t ti = e->type_index();
    if (ti == HAddObj::RuntimeTypeIndex()) {
      Visit(e.as<HAddObj>()->a), Visit(e.as<HAddObj>()->b);
    } else if (ti == HMulObj::RuntimeTypeIndex()) {
      Visit(e.as<HMulObj>()->a), Visit(e.as<HMulObj>()->b);
    } else if (ti == HFloorDivObj::RuntimeTypeIndex()) {
      Visit(e.as<HFloorDivObj>()->a), Visit(e.as<HFloorDivObj>()->b);
    } else if (ti == HFloorModObj::RuntimeTypeIndex()) {
      Visit(e.as<HFloorModObj>()->a), Visit(e.as<HFloorModObj>()->b);
    }
    ++visits_;
    cb_(e);
  }
  size_t visits() const { return visits_; }

 private:
  std::function<void(const HExpr&)> cb_;
  std::unordered_set<const Object*> visited_;
  size_t visits_ = 0;
};

class OldMutator {
 public:
  explicit OldMutator(std::function<Optional<HExpr>(const HVar&)> cb) : cb_(std::move(cb)) {}
  HExpr Mutate(const HExpr& e) {
    ++visits_;
    const int32_t ti = e->type_index();
    if (ti == HVarObj::RuntimeTypeIndex()) {
      if (Optional<HExpr> r = cb_(GetRef<HVar>(e.as<HVarObj>()))) return r.value();
      return e;
    }
    if (ti == HAddObj::RuntimeTypeIndex()) return Rebuild<HAddObj>(e);
    if (ti == HMulObj::RuntimeTypeIndex()) return Rebuild<HMulObj>(e);
    if (ti == HFloorDivObj::RuntimeTypeIndex()) return Rebuild<HFloorDivObj>(e);
    if (ti == HFloorModObj::RuntimeTypeIndex()) return Rebuild<HFloorModObj>(e);
    return e;
  }
  size_t visits() const { return visits_; }

 private:
  template <typename T>
  HExpr Rebuild(const HExpr& e) {
    const T* op = e.as<T>();
    HExpr a = Mutate(op->a), b = Mutate(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) return e;
    ObjectPtr<T> copy = make_object<T>(*op);
    copy->a = std::move(a);
    copy->b = std::move(b);
    return HExpr(std::move(copy));
  }
  std::function<Optional<HExpr>(const HVar&)> cb_;
  size_t visits_ = 0;
};

double MedianNs(int repeats, const std::function<void()>& work) {
  std::vector<double> samples;
  work();
  for (int s = 0; s < 9; ++s) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) work();
    auto ns =
        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    samples.push_back(ns / repeats);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace

int main() {
  Any ty = Any(HPrimType(make_object<HPrimTypeObj>(32)));
  HVar outer(make_object<HVarObj>(ty, 0));
  HVar inner(make_object<HVarObj>(ty, 1));
  HVar repl(make_object<HVarObj>(ty, 2));
  auto imm = [&](int64_t v) { return HExpr(make_object<HIntImmObj>(ty, v)); };
  // floordiv(q, 32) * 32 + floormod(r, 32),  q = outer * 16 + inner
  auto split_fuse = [&](bool share) {
    HExpr q = Bin<HAddObj>(ty, Bin<HMulObj>(ty, outer, imm(16)), inner);
    HExpr r = share ? q : Bin<HAddObj>(ty, Bin<HMulObj>(ty, outer, imm(16)), inner);
    return Bin<HAddObj>(ty, Bin<HMulObj>(ty, Bin<HFloorDivObj>(ty, q, imm(32)), imm(32)),
                        Bin<HFloorModObj>(ty, r, imm(32)));
  };
  // The same expression with every binary node replaced by Call(a, b): identical arity and shape,
  // operands in an Array, so the engine's seq-container path runs instead of fixed fields.
  auto split_fuse_call = [&](bool share) {
    HExpr q = Call2(ty, Call2(ty, outer, imm(16)), inner);
    HExpr r = share ? q : Call2(ty, Call2(ty, outer, imm(16)), inner);
    return Call2(ty, Call2(ty, Call2(ty, q, imm(32)), imm(32)), Call2(ty, r, imm(32)));
  };

  size_t sink = 0;
  auto run = [&](const char* name, const HExpr& root) {
    // Pin every node. Without a second reference the graph can reach refcount one, the first map
    // takes the in-place path and rewrites it, and every later replace sample is a no-op.
    std::vector<HExpr> pin;
    OldVisitor retain([&](const HExpr& n) { pin.push_back(n); });
    retain.Visit(root);

    auto replace = [&](const HVar& v) -> Optional<HExpr> {
      return v->id == 0 ? Optional<HExpr>(repl) : Optional<HExpr>(std::nullopt);
    };

    // Instrumented visit counts, one per path. Never assumed.
    size_t v_old_walk = 0, v_old_map = 0, v_new = 0;
    {
      OldVisitor c([](const HExpr&) {});
      c.Visit(root);
      v_old_walk = c.visits();
    }
    {
      OldMutator m(replace);
      m.Mutate(root);
      v_old_map = m.visits();
    }
    {
      StructuralWalk<WalkOrder::kPostOrder>(AnyView(root),
                                            [&](const HExpr&) -> Expected<WalkResult> {
                                              ++v_new;
                                              return WalkResult::Advance();
                                            });
    }

    double t_old_walk = MedianNs(20000, [&] {
      size_t h = 0;
      OldVisitor v([&](const HExpr& n) { h += n.as<HVarObj>() != nullptr; });
      v.Visit(root);
      sink += h;
    });
    double t_walk_var = MedianNs(20000, [&] {
      size_t h = 0;
      StructuralWalk<WalkOrder::kPostOrder>(AnyView(root),
                                            [&](const HVar&) -> Expected<WalkResult> {
                                              ++h;
                                              return WalkResult::Advance();
                                            });
      sink += h;
    });
    double t_walk_never = MedianNs(20000, [&] {
      size_t h = 0;
      StructuralWalk<WalkOrder::kPostOrder>(AnyView(root),
                                            [&](const HNever&) -> Expected<WalkResult> {
                                              ++h;
                                              return WalkResult::Advance();
                                            });
      sink += h;
    });
    double t_old_map =
        MedianNs(20000, [&] { sink += OldMutator(replace).Mutate(root)->type_index(); });
    double t_map_var = MedianNs(20000, [&] {
      sink += StructuralMap<WalkOrder::kPostOrder>(AnyView(root), [&](const HVar& v) -> Any {
                return v->id == 0 ? Any(repl) : Any(v);
              }).type_index();
    });
    double t_map_never = MedianNs(20000, [&] {
      sink += StructuralMap<WalkOrder::kPostOrder>(AnyView(root), [](const HNever& n) -> Any {
                return Any(n);
              }).type_index();
    });

    // The replace must change the graph on both paths, or the map arms measure nothing.
    HExpr old_out = OldMutator(replace).Mutate(root);
    Any new_out = StructuralMap<WalkOrder::kPostOrder>(
        AnyView(root), [&](const HVar& v) -> Any { return v->id == 0 ? Any(repl) : Any(v); });
    printf("%-10s old_visits(walk/map)=%zu/%zu new_visits=%zu changed(old/new)=%d/%d\n", name,
           v_old_walk, v_old_map, v_new, (int)!old_out.same_as(root),
           (int)!new_out.cast<HExpr>().same_as(root));
    printf("  %-12s %10s %12s %12s\n", "path", "ns/trav", "visits", "ns/visit");
    auto row = [](const char* p, double t, size_t v) {
      printf("  %-12s %10.1f %12zu %12.2f\n", p, t, v, t / v);
    };
    row("old walk", t_old_walk, v_old_walk);
    row("walk Var", t_walk_var, v_new);
    row("walk Never", t_walk_never, v_new);
    row("old map", t_old_map, v_old_map);
    row("map Var", t_map_var, v_new);
    row("map Never", t_map_never, v_new);
  };

  run("shared", split_fuse(true));
  run("distinct", split_fuse(false));
  printf("sink=%zu\n", sink);
  return 0;
}
