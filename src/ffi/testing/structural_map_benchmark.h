/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_FFI_SRC_FFI_TESTING_STRUCTURAL_MAP_BENCHMARK_H_
#define TVM_FFI_SRC_FFI_TESTING_STRUCTURAL_MAP_BENCHMARK_H_

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace tvm::ffi::testing {

class TestExprObj : public Object {
 public:
  // Match ir::ExprNode's two deliberately unvisited ObjectRef-sized fields.
  ObjectRef span;
  ObjectRef ty;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr uint32_t _type_child_slots = 8;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.structural_map.TestExpr", TestExprObj, Object);
};

class TestExpr : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestExpr, ObjectRef, TestExprObj);
};

inline ObjectRef BenchmarkPrimType() {
  static const ObjectRef type_object(make_object<TestExprObj>());
  return type_object;
}

class TestVarObj : public TestExprObj {
 public:
  String name;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.structural_map.TestVar", TestVarObj, TestExprObj);
};

class TestVar : public TestExpr {
 public:
  explicit TestVar(String name) {
    auto node = make_object<TestVarObj>();
    node->name = std::move(name);
    node->ty = BenchmarkPrimType();
    data_ = std::move(node);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestVar, TestExpr, TestVarObj);
};

class TestIntImmObj : public TestExprObj {
 public:
  int64_t value;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.structural_map.TestIntImm", TestIntImmObj,
                                    TestExprObj);
};

class TestIntImm : public TestExpr {
 public:
  explicit TestIntImm(int64_t value) {
    auto node = make_object<TestIntImmObj>();
    node->value = value;
    node->ty = BenchmarkPrimType();
    data_ = std::move(node);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestIntImm, TestExpr, TestIntImmObj);
};

template <typename TNode>
class TestBinaryObj : public TestExprObj {
 public:
  TestExpr a;
  TestExpr b;

  static constexpr bool _type_final = true;
  static constexpr uint32_t _type_child_slots = 0;
};

#define TVM_FFI_TEST_BINARY(Name)                                                           \
  class Test##Name##Obj : public TestBinaryObj<Test##Name##Obj> {                           \
   public:                                                                                  \
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.structural_map.Test" #Name, Test##Name##Obj, \
                                      TestExprObj);                                         \
  };                                                                                        \
  class Test##Name : public TestExpr {                                                      \
   public:                                                                                  \
    Test##Name(TestExpr a, TestExpr b) {                                                    \
      auto node = make_object<Test##Name##Obj>();                                           \
      node->a = std::move(a);                                                               \
      node->b = std::move(b);                                                               \
      node->ty = BenchmarkPrimType();                                                       \
      data_ = std::move(node);                                                              \
    }                                                                                       \
    TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Test##Name, TestExpr, Test##Name##Obj);      \
  }

TVM_FFI_TEST_BINARY(Add);
TVM_FFI_TEST_BINARY(Mul);
TVM_FFI_TEST_BINARY(FloorDiv);
TVM_FFI_TEST_BINARY(FloorMod);
#undef TVM_FFI_TEST_BINARY

inline TVMFFIAny VisitLeaf(StructuralVisitorObj*, AnyView) noexcept {
  return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Optional<VisitInterrupt>>(std::nullopt));
}

template <typename TNode>
TVMFFIAny VisitBinary(StructuralVisitorObj* visitor, AnyView value) noexcept {
  const auto* self = value.cast<const TNode*>();
  auto a_result = visitor->VisitExpected(self->a);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(a_result, AnyView(self->a));
  auto b_result = visitor->VisitExpected(self->b);
  return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(b_result));
}

inline TVMFFIAny MutateLeaf(StructuralMutatorObj*, AnyView value) noexcept {
  return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
}

template <typename TNode>
TVMFFIAny MutateBinary(StructuralMutatorObj* mutator, AnyView value) noexcept {
  try {
    const auto* self = value.cast<const TNode*>();
    Expected<Any> a_result = mutator->MutateExpected(self->a);
    if (TVM_FFI_PREDICT_FALSE(a_result.is_err())) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(a_result));
    }
    Expected<Any> b_result = mutator->MutateExpected(self->b);
    if (TVM_FFI_PREDICT_FALSE(b_result.is_err())) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(b_result));
    }
    TestExpr a = std::move(details::ExpectedUnsafe::GetData(a_result)).cast<TestExpr>();
    TestExpr b = std::move(details::ExpectedUnsafe::GetData(b_result)).cast<TestExpr>();
    if (a.same_as(self->a) && b.same_as(self->b)) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
    }
    auto copy = make_object<TNode>(*self);
    copy->a = std::move(a);
    copy->b = std::move(b);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(ObjectRef(std::move(copy)))));
  } catch (const Error& err) {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Unexpected(err)));
  }
}

template <typename TNode>
TVMFFIAny MaybeInplaceMutateBinary(StructuralMutatorObj* mutator, AnyView value) noexcept {
  try {
    auto* self = const_cast<TNode*>(value.cast<const TNode*>());
    Expected<Any> a_result = mutator->MaybeInplaceMutateIfUniqueExpected(self->a);
    if (TVM_FFI_PREDICT_FALSE(a_result.is_err())) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(a_result));
    }
    Expected<Any> b_result = mutator->MaybeInplaceMutateIfUniqueExpected(self->b);
    if (TVM_FFI_PREDICT_FALSE(b_result.is_err())) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(b_result));
    }
    TestExpr a = std::move(details::ExpectedUnsafe::GetData(a_result)).cast<TestExpr>();
    TestExpr b = std::move(details::ExpectedUnsafe::GetData(b_result)).cast<TestExpr>();
    if (!a.same_as(self->a)) self->a = std::move(a);
    if (!b.same_as(self->b)) self->b = std::move(b);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Any(value)));
  } catch (const Error& err) {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Expected<Any>(Unexpected(err)));
  }
}

inline void RegisterStructuralMapBenchmarkTypes() {
  static bool registered = [] {
    namespace refl = reflection;
    refl::ObjectDef<TestExprObj>()
        .def_ro("span", &TestExprObj::span, refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("ty", &TestExprObj::ty, refl::AttachFieldFlag::SEqHashIgnore());
    refl::ObjectDef<TestVarObj>().def_ro("name", &TestVarObj::name,
                                         refl::AttachFieldFlag::SEqHashIgnore());
    refl::ObjectDef<TestIntImmObj>().def_ro("value", &TestIntImmObj::value);
    refl::ObjectDef<TestAddObj>().def_ro("a", &TestAddObj::a).def_ro("b", &TestAddObj::b);
    refl::ObjectDef<TestMulObj>().def_ro("a", &TestMulObj::a).def_ro("b", &TestMulObj::b);
    refl::ObjectDef<TestFloorDivObj>()
        .def_ro("a", &TestFloorDivObj::a)
        .def_ro("b", &TestFloorDivObj::b);
    refl::ObjectDef<TestFloorModObj>()
        .def_ro("a", &TestFloorModObj::a)
        .def_ro("b", &TestFloorModObj::b);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMaybeInplaceMutate);
#define TVM_FFI_REGISTER_HOOKS(Type, Visit, Mutate, Maybe)                       \
  refl::TypeAttrDef<Type>()                                                      \
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(Visit))   \
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(Mutate)) \
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate, reinterpret_cast<void*>(Maybe))
    TVM_FFI_REGISTER_HOOKS(TestVarObj, &VisitLeaf, &MutateLeaf, &MutateLeaf);
    TVM_FFI_REGISTER_HOOKS(TestIntImmObj, &VisitLeaf, &MutateLeaf, &MutateLeaf);
    TVM_FFI_REGISTER_HOOKS(TestAddObj, &VisitBinary<TestAddObj>, &MutateBinary<TestAddObj>,
                           &MaybeInplaceMutateBinary<TestAddObj>);
    TVM_FFI_REGISTER_HOOKS(TestMulObj, &VisitBinary<TestMulObj>, &MutateBinary<TestMulObj>,
                           &MaybeInplaceMutateBinary<TestMulObj>);
    TVM_FFI_REGISTER_HOOKS(TestFloorDivObj, &VisitBinary<TestFloorDivObj>,
                           &MutateBinary<TestFloorDivObj>,
                           &MaybeInplaceMutateBinary<TestFloorDivObj>);
    TVM_FFI_REGISTER_HOOKS(TestFloorModObj, &VisitBinary<TestFloorModObj>,
                           &MutateBinary<TestFloorModObj>,
                           &MaybeInplaceMutateBinary<TestFloorModObj>);
#undef TVM_FFI_REGISTER_HOOKS
    return true;
  }();
  (void)registered;
}

}  // namespace tvm::ffi::testing

#endif  // TVM_FFI_SRC_FFI_TESTING_STRUCTURAL_MAP_BENCHMARK_H_
