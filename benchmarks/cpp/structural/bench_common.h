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
#ifndef TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_BENCH_COMMON_H_
#define TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_BENCH_COMMON_H_

// Shared core of the structural-traversal benchmark harnesses.
//
// The two harnesses -- mini-TIR (`mini_tir_bench.cc`, tvm-ffi types only) and real TVM
// (`real_tvm_bench.cc`, links apache/tvm) -- differ only in the node types they build from
// and in how structural hooks are installed.  Everything a report depends on lives here:
// the arm vocabulary, the timing method, the working-set/cache accounting, the assertions
// that fail a run, and the machine-readable emission format the reporter renders.
//
// Nothing here knows a single concrete node type.

#include <tvm/ffi/any.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>

#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {
namespace bench {

// Emission primitives, defined at the bottom of this header.
inline void Emit(const std::string& line);
inline void EmitProvenance(const std::string& key, const std::string& value);

// ---------------------------------------------------------------------------
// Arm vocabulary -- defined once, selected (never redefined) by a measurement.
// ---------------------------------------------------------------------------

/*! \brief Which quantity an arm measures. One table per quantity, never mixed. */
enum class Quantity { kWalk, kMap };

/*! \brief The fixed arm vocabulary. A new measurement selects from this list. */
enum class ArmId {
  // walk quantity -- no ownership axis, a walk returns a status and owns nothing.
  kWalkFloor,   // minimal hand-written vtable: attr lookup + hook dispatch, no callbacks
  kWalk,        // StructuralWalk<kPostOrder> with a Var link and an Expr catch-all link
  kWalkNever,   // same shape, first link can never match: prices link testing alone
  kWalkOld,     // the shipping traversal this replaces (PostOrderVisit / model)
  // map quantity -- every arm runs in both ownership variants.
  kMapFloor,     // minimal hand-written vtable mutator: the engine's lower bound
  kMapNever,     // StructuralMap whose callback can never match
  kMapIdentity,  // StructuralMap<Var> returning the same Var: matches, rebuilds nothing
  kMapReplace,   // StructuralMap<Var> replacing one variable: matches and rebuilds
  kMapOld,       // the shipping mutation this replaces (Substitute / model)
};

/*! \brief Ownership of the root handed to a map arm. Walk arms have no ownership axis. */
enum class Ownership {
  kRetained,  // the caller still holds the root: refcount >= 2, copy-on-write
  kMoved,     // std::move(root): uniquely owned, the in-place path, nothing rebuilds
};

/*! \brief Static description of one arm. */
struct ArmSpec {
  ArmId id;
  const char* name;
  Quantity quantity;
  bool has_ownership_axis;
  bool rebuilds;  // whether the arm changes the graph at all
  const char* doc;
};

/*! \brief The vocabulary itself. */
inline const std::vector<ArmSpec>& Arms() {
  static const std::vector<ArmSpec> arms = {
      {ArmId::kWalkFloor, "walk_floor", Quantity::kWalk, false, false,
       "minimal hand-written vtable walk: type-attr lookup and hook dispatch, no callbacks"},
      {ArmId::kWalk, "walk", Quantity::kWalk, false, false,
       "StructuralWalk<kPostOrder> with a Var link and an Expr catch-all link"},
      {ArmId::kWalkNever, "walk_never", Quantity::kWalk, false, false,
       "StructuralWalk<kPostOrder> whose first link can never match"},
      {ArmId::kWalkOld, "walk_old", Quantity::kWalk, false, false,
       "the shipping post-order traversal this replaces"},
      {ArmId::kMapFloor, "map_floor", Quantity::kMap, true, false,
       "minimal hand-written vtable mutator: the engine's lower bound"},
      {ArmId::kMapNever, "map_never", Quantity::kMap, true, false,
       "StructuralMap<kPostOrder> whose callback can never match"},
      {ArmId::kMapIdentity, "map_identity", Quantity::kMap, true, false,
       "StructuralMap<kPostOrder> over Var returning the same Var: matches, rebuilds nothing"},
      {ArmId::kMapReplace, "map_replace", Quantity::kMap, true, true,
       "StructuralMap<kPostOrder> over Var replacing one variable"},
      {ArmId::kMapOld, "map_old", Quantity::kMap, true, true,
       "the shipping substitution this replaces"},
  };
  return arms;
}

inline const ArmSpec& ArmSpecOf(ArmId id) {
  for (const ArmSpec& spec : Arms()) {
    if (spec.id == id) return spec;
  }
  std::abort();
}

inline const char* OwnershipName(Ownership o) {
  return o == Ownership::kRetained ? "retained" : "moved";
}

// ---------------------------------------------------------------------------
// Method -- fixed here, never restated per report.
// ---------------------------------------------------------------------------

struct Method {
  /*! \brief One untimed warm-up batch before any sample. */
  static constexpr int kWarmupBatches = 1;
  /*! \brief Timed samples per arm per process. The process reports their median. */
  static constexpr int kSamples = 9;
  /*! \brief Target wall time of one sample; the repeat count is calibrated to it. */
  static constexpr double kTargetSampleNs = 30e6;
  /*! \brief Bounds on the calibrated repeat count. */
  static constexpr int kMinRepeats = 8;
  static constexpr int kMaxRepeats = 20000;
  /*! \brief A timed batch is sized to at least this, so its two clock reads do not show. */
  static constexpr double kMinBatchNs = 200e3;
  static constexpr int kMaxBatch = 20000;
  /*! \brief Independent pinned processes whose medians the reporter medians again. */
  static constexpr int kProcessRuns = 5;
  static constexpr const char* kDescription =
      "one untimed warm-up batch, 9 timed samples per arm per process, batch sized to >=200 us "
      "and repeats calibrated to ~30 ms per sample, process value = median of its 9 samples, "
      "reported value = median of 5 pinned process medians";
};

// ---------------------------------------------------------------------------
// Machine facts: cache geometry and the level a working set fits in.
// ---------------------------------------------------------------------------

struct CacheInfo {
  int64_t l1d = 0;
  int64_t l2 = 0;
  int64_t l3 = 0;
};

inline int64_t ReadSysfsSize(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) return 0;
  char buf[64] = {0};
  if (std::fgets(buf, sizeof(buf), f) == nullptr) {
    std::fclose(f);
    return 0;
  }
  std::fclose(f);
  int64_t value = std::atoll(buf);
  if (std::strchr(buf, 'K') != nullptr) value *= 1024;
  if (std::strchr(buf, 'M') != nullptr) value *= 1024 * 1024;
  return value;
}

inline std::string ReadSysfsString(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (f == nullptr) return "";
  char buf[64] = {0};
  if (std::fgets(buf, sizeof(buf), f) == nullptr) {
    std::fclose(f);
    return "";
  }
  std::fclose(f);
  std::string s(buf);
  while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
  return s;
}

/*! \brief Read the running CPU's data-cache geometry from sysfs. */
inline CacheInfo DetectCaches() {
  CacheInfo info;
  for (int index = 0; index < 8; ++index) {
    std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";
    std::string type = ReadSysfsString(base + "type");
    if (type.empty()) continue;
    int level = static_cast<int>(ReadSysfsSize(base + "level"));
    int64_t size = ReadSysfsSize(base + "size");
    if (type == "Instruction") continue;
    if (level == 1) info.l1d = size;
    if (level == 2) info.l2 = size;
    if (level == 3) info.l3 = size;
  }
  return info;
}

/*!
 * \brief Name the smallest cache level the working set fits in.
 *
 * Reported rather than asserted: the whole point of the SeqStmt sweep is that the benefit
 * of removing refcount and ALU work should decay as the fixture stops fitting, and that has
 * to be visible in the table instead of carried as prose.
 */
inline std::string FitsLevel(int64_t working_set, const CacheInfo& cache) {
  if (cache.l1d > 0 && working_set <= cache.l1d) return "L1d";
  if (cache.l2 > 0 && working_set <= cache.l2) return "L2";
  if (cache.l3 > 0 && working_set <= cache.l3) return "L3";
  return "DRAM";
}

/*! \brief The CPUs this process is pinned to, as a compact string. */
inline std::string PinningDescription() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) return "unknown";
  std::string out;
  int total = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) {
      ++total;
      if (total <= 4) out += (out.empty() ? "" : ",") + std::to_string(cpu);
    }
  }
  if (total > 4) out += ",... (" + std::to_string(total) + " cpus, not pinned)";
  return out.empty() ? "unknown" : out;
}

/*! \brief The `model name` line from /proc/cpuinfo. */
inline std::string CpuModel() {
  FILE* f = std::fopen("/proc/cpuinfo", "r");
  if (f == nullptr) return "unknown";
  char line[512];
  std::string model = "unknown";
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "model name", 10) == 0) {
      const char* colon = std::strchr(line, ':');
      if (colon != nullptr) {
        model = colon + 2;
        while (!model.empty() && (model.back() == '\n' || model.back() == ' ')) model.pop_back();
      }
      break;
    }
  }
  std::fclose(f);
  return model;
}

#ifndef TVM_FFI_BENCH_HARNESS_COMMIT
#define TVM_FFI_BENCH_HARNESS_COMMIT "unknown"
#endif
#ifndef TVM_FFI_BENCH_ENGINE_SHA
#define TVM_FFI_BENCH_ENGINE_SHA "unknown"
#endif
#ifndef TVM_FFI_BENCH_TVM_SHA
#define TVM_FFI_BENCH_TVM_SHA "unknown"
#endif
#ifndef TVM_FFI_BENCH_TVM_FFI_PIN
#define TVM_FFI_BENCH_TVM_FFI_PIN "n/a"
#endif
#ifndef TVM_FFI_BENCH_CXX_FLAGS
#define TVM_FFI_BENCH_CXX_FLAGS "unknown"
#endif
#ifndef TVM_FFI_BENCH_GUARD_DISABLED
#define TVM_FFI_BENCH_GUARD_DISABLED "unknown"
#endif

/*!
 * \brief Emit the provenance block.
 *
 * Printed by the harness rather than remembered by the author: a report cannot omit what
 * the tool prints.
 */
inline void EmitStandardProvenance(const std::string& harness) {
  const CacheInfo cache = DetectCaches();
  EmitProvenance("harness", harness);
  EmitProvenance("harness_branch_commit", TVM_FFI_BENCH_HARNESS_COMMIT);
  EmitProvenance("tvm_ffi_engine_sha", TVM_FFI_BENCH_ENGINE_SHA);
  EmitProvenance("tvm_sha", TVM_FFI_BENCH_TVM_SHA);
  EmitProvenance("tvm_ffi_submodule_pin", TVM_FFI_BENCH_TVM_FFI_PIN);
  EmitProvenance("machine", CpuModel());
  EmitProvenance("compiler",
#if defined(__clang__)
                 std::string("clang ") + __clang_version__
#elif defined(__GNUC__)
                 std::string("gcc ") + __VERSION__
#else
                 std::string("unknown")
#endif
  );
  EmitProvenance("flags", TVM_FFI_BENCH_CXX_FLAGS);
  EmitProvenance("write_once_attr_guard", TVM_FFI_BENCH_GUARD_DISABLED);
  EmitProvenance("pinning", PinningDescription());
  EmitProvenance("cache_l1d_bytes", std::to_string(cache.l1d));
  EmitProvenance("cache_l2_bytes", std::to_string(cache.l2));
  EmitProvenance("cache_l3_bytes", std::to_string(cache.l3));
  EmitProvenance("method", Method::kDescription);
}

// ---------------------------------------------------------------------------
// Graph facts: unique nodes, occurrences, working set, rebuild counts.
// ---------------------------------------------------------------------------

/*!
 * \brief Size in bytes of one node, or 0 when the harness does not know the type.
 *
 * Takes the object too, so a container whose payload is inline (an `Array`) can report its
 * real footprint rather than the size of its header.
 */
using NodeSizeFn = std::function<int64_t(int32_t, const Object*)>;

struct GraphStats {
  int64_t unique_nodes = 0;
  int64_t occurrences = 0;
  int64_t working_set = 0;
  int64_t unsized_nodes = 0;
  std::unordered_set<const Object*> nodes;
};

/*!
 * \brief Walk \p root and collect its unique object identities and working-set size.
 *
 * Uses the same engine the arms use, with an `AnyView` catch-all link, so "unique nodes"
 * is exactly the set of objects a traversal touches -- not a hand-maintained constant.
 */
inline GraphStats CollectGraph(AnyView root, const NodeSizeFn& size_of) {
  GraphStats stats;
  StructuralWalk<WalkOrder::kPostOrder>(root, [&](AnyView value) -> Expected<WalkResult> {
    if (value.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      TVMFFIAny raw = value.CopyToTVMFFIAny();
      const Object* obj = details::ObjectUnsafe::RawObjectPtrFromUnowned<Object>(raw.v_obj);
      ++stats.occurrences;
      if (stats.nodes.insert(obj).second) {
        ++stats.unique_nodes;
        int64_t bytes = size_of(value.type_index(), obj);
        if (bytes == 0) {
          ++stats.unsized_nodes;
        } else {
          stats.working_set += bytes;
        }
      }
    }
    return WalkResult::Advance();
  });
  return stats;
}

/*! \brief Number of identities in \p after that are absent from \p before. */
inline int64_t CountRebuilt(AnyView before, AnyView after, const NodeSizeFn& size_of) {
  GraphStats a = CollectGraph(before, size_of);
  GraphStats b = CollectGraph(after, size_of);
  int64_t rebuilt = 0;
  for (const Object* obj : b.nodes) rebuilt += a.nodes.count(obj) == 0;
  return rebuilt;
}

// ---------------------------------------------------------------------------
// Dispatch counting: a trampoline over every registered structural hook.
// ---------------------------------------------------------------------------

/*!
 * \brief Counts how many registered structural-hook dispatches an arm performs.
 *
 * Installed by re-registering `__s_visit__` / `__s_mutate__` / `__s_maybe_inplace_mutate__`
 * over the hooks already in the type-attribute columns, which is what the branch-local
 * `TVM_FFI_BENCH_ALLOW_TYPE_ATTR_OVERRIDE` build exists for.  The trampolines are installed
 * for the untimed assertion pass only and removed again before anything is timed, so no
 * reported number carries their cost.
 *
 * This is the arm-agnostic form of the dispatch-count check: the counter sits under the
 * engine rather than inside one arm's callbacks, so two arms that claim to traverse the
 * same nodes are compared on the same observable.
 */
class DispatchCounter {
 public:
  static DispatchCounter* Global() {
    static DispatchCounter inst;
    return &inst;
  }

  /*!
   * \brief Wrap every hook registered for \p type_indices.
   * \return false when the linked tvm-ffi still enforces the write-once type-attribute
   *         guard, in which case no hook was replaced and dispatch counting is unavailable.
   */
  bool Install(const std::vector<int32_t>& type_indices);
  /*! \brief Put the original hooks back. */
  void Restore();
  bool installed() const { return installed_; }

  int64_t count = 0;

 private:
  struct Saved {
    int32_t type_index;
    void* visit = nullptr;
    void* mutate = nullptr;
    void* inplace = nullptr;
  };

  static TVMFFIAny VisitTrampoline(StructuralVisitorObj* self, AnyView value) noexcept {
    DispatchCounter* c = Global();
    ++c->count;
    auto it = c->by_index_.find(value.type_index());
    return (*reinterpret_cast<FStructuralVisit>(it->second.visit))(self, value);
  }
  static TVMFFIAny MutateTrampoline(StructuralMutatorObj* self, AnyView value) noexcept {
    DispatchCounter* c = Global();
    ++c->count;
    auto it = c->by_index_.find(value.type_index());
    return (*reinterpret_cast<FStructuralMutate>(it->second.mutate))(self, value);
  }
  static TVMFFIAny InplaceTrampoline(StructuralMutatorObj* self, AnyView value) noexcept {
    DispatchCounter* c = Global();
    ++c->count;
    auto it = c->by_index_.find(value.type_index());
    return (*reinterpret_cast<FStructuralMutate>(it->second.inplace))(self, value);
  }

  bool installed_ = false;
  std::unordered_map<int32_t, Saved> by_index_;
};

namespace detail {
inline void* AttrPtr(const reflection::TypeAttrColumn& column, int32_t type_index) {
  AnyView attr = column[type_index];
  if (attr.type_index() != TypeIndex::kTVMFFIOpaquePtr) return nullptr;
  return attr.cast<void*>();
}
inline void SetAttr(int32_t type_index, const char* name, void* fn) {
  TVMFFIByteArray name_array{name, std::strlen(name)};
  TVMFFIAny value_any = AnyView(fn).CopyToTVMFFIAny();
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index, &name_array, &value_any));
}
}  // namespace detail

inline bool DispatchCounter::Install(const std::vector<int32_t>& type_indices) {
  reflection::TypeAttrColumn visit_column(reflection::type_attr::kStructuralVisit);
  reflection::TypeAttrColumn mutate_column(reflection::type_attr::kStructuralMutate);
  reflection::TypeAttrColumn inplace_column(reflection::type_attr::kStructuralMaybeInplaceMutate);
  by_index_.clear();
  for (int32_t type_index : type_indices) {
    Saved saved;
    saved.type_index = type_index;
    saved.visit = detail::AttrPtr(visit_column, type_index);
    saved.mutate = detail::AttrPtr(mutate_column, type_index);
    saved.inplace = detail::AttrPtr(inplace_column, type_index);
    if (saved.visit == nullptr && saved.mutate == nullptr && saved.inplace == nullptr) continue;
    by_index_[type_index] = saved;
  }
  try {
    for (const auto& kv : by_index_) {
      if (kv.second.visit != nullptr) {
        detail::SetAttr(kv.first, reflection::type_attr::kStructuralVisit,
                        reinterpret_cast<void*>(&DispatchCounter::VisitTrampoline));
      }
      if (kv.second.mutate != nullptr) {
        detail::SetAttr(kv.first, reflection::type_attr::kStructuralMutate,
                        reinterpret_cast<void*>(&DispatchCounter::MutateTrampoline));
      }
      if (kv.second.inplace != nullptr) {
        detail::SetAttr(kv.first, reflection::type_attr::kStructuralMaybeInplaceMutate,
                        reinterpret_cast<void*>(&DispatchCounter::InplaceTrampoline));
      }
    }
  } catch (const Error&) {
    // The linked tvm-ffi still enforces the write-once guard: nothing to restore, because
    // the very first replacement is the one that threw.
    by_index_.clear();
    return false;
  }
  installed_ = true;
  return true;
}

inline void DispatchCounter::Restore() {
  if (!installed_) return;
  for (const auto& kv : by_index_) {
    if (kv.second.visit != nullptr) {
      detail::SetAttr(kv.first, reflection::type_attr::kStructuralVisit, kv.second.visit);
    }
    if (kv.second.mutate != nullptr) {
      detail::SetAttr(kv.first, reflection::type_attr::kStructuralMutate, kv.second.mutate);
    }
    if (kv.second.inplace != nullptr) {
      detail::SetAttr(kv.first, reflection::type_attr::kStructuralMaybeInplaceMutate,
                      kv.second.inplace);
    }
  }
  installed_ = false;
}

// ---------------------------------------------------------------------------
// Floor arms: minimal hand-written vtables. Type-agnostic, so they live here.
// ---------------------------------------------------------------------------

/*! \brief Lower bound for a hooked walk: attr-column lookup and hook dispatch, no callbacks. */
class MinimalVisitorObj final : public StructuralVisitorObj {
 public:
  MinimalVisitorObj() : StructuralVisitorObj(VTable()) {}

 private:
  static TVMFFIAny Visit(StructuralVisitorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralVisit);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralVisit>(attr.cast<void*>()))(self, value);
    }
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable vtable{&MinimalVisitorObj::Visit};
    return &vtable;
  }
};

/*! \brief Lower bound for a hooked map: attr-column lookup and hook dispatch, no callbacks. */
class MinimalMutatorObj final : public StructuralMutatorObj {
 public:
  MinimalMutatorObj() : StructuralMutatorObj(VTable()) {}

 private:
  static TVMFFIAny Mutate(StructuralMutatorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMutate);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(self, value);
    }
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  static TVMFFIAny InplaceMutate(StructuralMutatorObj* self, AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMaybeInplaceMutate);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      return (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(self, value);
    }
    return Mutate(self, value);
  }
  static TVMFFIAny NoRemapGet(StructuralMutatorObj*, AnyView) noexcept {
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static TVMFFIAny NoRemapSet(StructuralMutatorObj*, AnyView, AnyView) noexcept {
    return AnyView(nullptr).CopyToTVMFFIAny();
  }
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{&MinimalMutatorObj::Mutate,
                                                &MinimalMutatorObj::InplaceMutate,
                                                &MinimalMutatorObj::NoRemapGet,
                                                &MinimalMutatorObj::NoRemapSet};
    return &vtable;
  }
};

// ---------------------------------------------------------------------------
// Timing.
// ---------------------------------------------------------------------------

/*!
 * \brief One timed arm binding.
 *
 * `prepare` runs untimed before every timed batch; `run(i)` performs traversal `i` of the
 * batch.  A map arm whose in-place path leaves the graph invariant uses a large batch and
 * an empty `prepare`.  An arm that consumes its fixture -- an in-place replacement on a
 * fixture with shared subtrees, which un-shares it -- uses a pool of independent copies,
 * with `batch` equal to the pool size and `prepare` rebuilding the pool.
 */
struct ArmBinding {
  ArmId id;
  Ownership ownership = Ownership::kRetained;
  /*!
   * \brief Traversals per timed batch.
   *
   * 0 means the arm is stationary -- repeating it leaves the fixture unchanged -- and the
   * harness sizes the batch itself so the two clock reads per batch stay far below the
   * measured quantity.  A pooled arm sets this to its pool size.
   */
  int batch = 0;
  std::function<void()> prepare = [] {};
  std::function<void(int)> run;
};

inline double NowNs() {
  return std::chrono::duration<double, std::nano>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/*! \brief Median nanoseconds per traversal for one arm, by the fixed method above. */
inline double MeasureNsPerTraversal(const ArmBinding& arm_in) {
  ArmBinding arm = arm_in;
  // Warm-up, untimed, and a first estimate of one traversal.
  arm.prepare();
  double probe_begin = NowNs();
  arm.run(0);
  double one = NowNs() - probe_begin;
  if (arm.batch == 0) {
    // Stationary: size the batch so the batch's two clock reads are noise against it.
    int batch = one > 0 ? static_cast<int>(Method::kMinBatchNs / one) : Method::kMaxBatch;
    arm.batch = std::max(1, std::min(Method::kMaxBatch, batch));
  }
  for (int w = 0; w < Method::kWarmupBatches; ++w) {
    arm.prepare();
    for (int i = 0; i < arm.batch; ++i) arm.run(i);
  }
  // Calibrate the repeat count so one sample lands near the target duration.
  double t0 = NowNs();
  arm.prepare();
  for (int i = 0; i < arm.batch; ++i) arm.run(i);
  double per_batch = NowNs() - t0;
  int repeats = 1;
  if (per_batch > 0) {
    repeats = static_cast<int>(Method::kTargetSampleNs / per_batch);
  }
  repeats = std::max(Method::kMinRepeats, std::min(Method::kMaxRepeats, repeats));

  std::vector<double> samples;
  samples.reserve(Method::kSamples);
  for (int sample = 0; sample < Method::kSamples; ++sample) {
    double total = 0;
    for (int r = 0; r < repeats; ++r) {
      arm.prepare();
      double begin = NowNs();
      for (int i = 0; i < arm.batch; ++i) arm.run(i);
      total += NowNs() - begin;
    }
    samples.push_back(total / (static_cast<double>(repeats) * arm.batch));
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// ---------------------------------------------------------------------------
// Emission. One line per record; the reporter parses these and renders tables.
// ---------------------------------------------------------------------------

inline void Emit(const std::string& line) { std::printf("%s\n", line.c_str()); }

inline void EmitProvenance(const std::string& key, const std::string& value) {
  Emit("#provenance\t" + key + "\t" + value);
}

inline void EmitFixture(const std::string& harness, const std::string& fixture,
                        const std::string& ownership, int64_t unique_nodes, int64_t occurrences,
                        int64_t working_set, const std::string& fits, int64_t changed,
                        int64_t unchanged) {
  Emit("#fixture\t" + harness + "\t" + fixture + "\t" + ownership + "\t" +
       std::to_string(unique_nodes) + "\t" + std::to_string(occurrences) + "\t" +
       std::to_string(working_set) + "\t" + fits + "\t" + std::to_string(changed) + "\t" +
       std::to_string(unchanged));
}

inline void EmitResult(const std::string& harness, const std::string& fixture,
                       const std::string& ownership, const std::string& arm, int64_t unique_nodes,
                       double ns_per_traversal) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4f", ns_per_traversal);
  Emit("#result\t" + harness + "\t" + fixture + "\t" + ownership + "\t" + arm + "\t" +
       std::to_string(unique_nodes) + "\t" + buf);
}

inline void EmitCounter(const std::string& harness, const std::string& fixture,
                        const std::string& ownership, const std::string& arm,
                        const std::string& counter, int64_t value) {
  Emit("#counter\t" + harness + "\t" + fixture + "\t" + ownership + "\t" + arm + "\t" + counter +
       "\t" + std::to_string(value));
}

/*!
 * \brief Record one assertion outcome.
 *
 * A failing assertion aborts the process before anything is timed, so a wrong measurement
 * stops rather than reaching a table.
 */
inline void EmitAssert(const std::string& harness, const std::string& name, bool ok,
                       const std::string& detail) {
  Emit("#assert\t" + harness + "\t" + name + "\t" + (ok ? "pass" : "FAIL") + "\t" + detail);
  if (!ok) {
    std::fflush(stdout);
    std::fprintf(stderr, "structural benchmark assertion failed: %s (%s)\n", name.c_str(),
                 detail.c_str());
    std::exit(2);
  }
}


// ---------------------------------------------------------------------------
// Fixtures and the shared per-fixture driver.
//
// Everything below is common to both harnesses.  A harness supplies a Policy with its own
// node types:
//
//   struct Policy {
//     static constexpr const char* kName;
//     static int64_t NodeSize(int32_t type_index, const Object* obj);
//     static void RunWalkArm(ArmId id, AnyView root);
//     static void RunMapArm(ArmId id, Ownership ownership, Any* slot);
//     static Any MapReplace(Any root);    // the arms again, but returning their output, so
//     static Any MapIdentity(Any root);   // the untimed assertion pass can inspect it
//     static Any MapNever(Any root);
//     static Any MapOld(Any root);
//     // whether `walk_old` / `map_old` are themselves built on the structural engine, which
//     // decides whether their hook-dispatch counts are comparable to the other arms'
//     static constexpr bool kOldGoesThroughEngine;
//   };
// ---------------------------------------------------------------------------

/*! \brief A named fixture, built fresh on demand so a pool of independent copies exists. */
struct Fixture {
  std::string name;
  std::function<Any()> build;
  // Fixtures with a pointer-shared subtree cannot be traversed in place repeatedly: the
  // first in-place rebuild un-shares the DAG. Those cases use a pool of fresh copies.
  bool has_sharing = false;
  // Declared expectations, asserted before anything is timed. -1 means "not declared":
  // the harness still reports the measured value, it just does not fail the run on it.
  int64_t expect_unique_nodes = -1;
  int64_t expect_rebuilt_retained = -1;
  int64_t expect_rebuilt_moved = -1;
  int64_t expect_changed = -1;  // nodes on the path from a replaced Var to the root
  // Occurrences of an already-remapped variable, which the identity remap serves without
  // entering a hook. Distinguishes "traverses fewer nodes" from "reuses a remapped answer".
  int64_t expect_remap_hits = -1;
};

/*!
 * \brief The `SeqStmt` length sweep, chosen to cross the cache boundaries.
 *
 * Both harnesses build the same lengths so their rows are structural counterparts, and the
 * harness reports which cache level each one actually lands in rather than asserting it.
 */
inline const std::vector<int>& SeqSweepLengths() {
  static const std::vector<int> lengths = {16, 64, 256, 1024, 4096, 16384};
  return lengths;
}


/*! \brief True while this build can override registered hooks. */
inline bool* CanCountDispatchesFlag() {
  static bool flag = true;
  return &flag;
}
inline bool CanCountDispatches() { return *CanCountDispatchesFlag(); }

/*! \brief Type indices whose hooks the dispatch counter wraps. */
std::vector<int32_t> HookedTypeIndices() {
  std::vector<int32_t> indices;
  for (int32_t index = 0; index < 4096; ++index) indices.push_back(index);
  return indices;
}

/*! \brief True when this build can override registered hooks (the guard-disabled build). */


/*!
 * \brief Count registered structural-hook dispatches performed by one arm.
 * \return -1 when the linked tvm-ffi enforces the write-once type-attribute guard, so no
 *         trampoline could be installed and the dispatch assertions cannot run.
 */
int64_t CountDispatches(const std::function<void()>& arm) {
  DispatchCounter* counter = DispatchCounter::Global();
  if (!counter->Install(HookedTypeIndices())) {
    *CanCountDispatchesFlag() = false;
    return -1;
  }
  counter->count = 0;
  arm();
  int64_t count = counter->count;
  counter->Restore();
  return count;
}

template <typename Policy>
void RunFixture(const Fixture& fixture) {
  const CacheInfo cache = DetectCaches();
  const NodeSizeFn size_of = &Policy::NodeSize;

  Any root = fixture.build();
  GraphStats stats = CollectGraph(root, size_of);

  EmitAssert(Policy::kName, "fixture/" + fixture.name + "/all-nodes-sized", stats.unsized_nodes == 0,
             "unsized=" + std::to_string(stats.unsized_nodes));
  EmitCounter(Policy::kName, fixture.name, "-", "fixture", "unique_nodes", stats.unique_nodes);
  EmitCounter(Policy::kName, fixture.name, "-", "fixture", "occurrences", stats.occurrences);
  if (fixture.expect_unique_nodes >= 0) {
    EmitAssert(Policy::kName, "fixture/" + fixture.name + "/unique-nodes",
               stats.unique_nodes == fixture.expect_unique_nodes,
               "measured=" + std::to_string(stats.unique_nodes) +
                   " declared=" + std::to_string(fixture.expect_unique_nodes));
  }

  const std::string fits = FitsLevel(stats.working_set, cache);

  // ---- rebuild and identity counts, per ownership variant -------------------
  {
    Any retained_input = fixture.build();
    Any retained_output = Policy::MapReplace(Any(retained_input));
    int64_t rebuilt = CountRebuilt(retained_input, retained_output, size_of);
    EmitCounter(Policy::kName, fixture.name, "retained", "map_replace", "rebuilt", rebuilt);
    EmitAssert(Policy::kName, "rebuilt/" + fixture.name + "/retained/map_replace",
               fixture.expect_rebuilt_retained < 0 || rebuilt == fixture.expect_rebuilt_retained,
               "measured=" + std::to_string(rebuilt) +
                   " declared=" + std::to_string(fixture.expect_rebuilt_retained));

    Any identity_output = Policy::MapIdentity(Any(retained_input));
    EmitAssert(Policy::kName, "identity/" + fixture.name + "/retained/map_identity",
               CountRebuilt(retained_input, identity_output, size_of) == 0,
               "map_identity must return the input graph unchanged");

    Any never_output = Policy::MapNever(Any(retained_input));
    EmitAssert(Policy::kName, "identity/" + fixture.name + "/retained/map_never",
               CountRebuilt(retained_input, never_output, size_of) == 0,
               "map_never must return the input graph unchanged");

    Any old_input = fixture.build();
    Any old_output = Policy::MapOld(Any(old_input));
    int64_t old_rebuilt = CountRebuilt(old_input, old_output, size_of);
    EmitCounter(Policy::kName, fixture.name, "retained", "map_old", "rebuilt", old_rebuilt);
    EmitAssert(Policy::kName, "rebuilt/" + fixture.name + "/retained/map_old-matches-map_replace",
               old_rebuilt == rebuilt,
               "map_old=" + std::to_string(old_rebuilt) + " map_replace=" + std::to_string(rebuilt));
  }
  {
    // Moved: the sole reference goes in, so nothing should be rebuilt except what a
    // pointer-shared subtree forces.
    Any moved = fixture.build();
    GraphStats before = CollectGraph(moved, size_of);
    Any after = Policy::MapReplace(std::move(moved));
    GraphStats after_stats = CollectGraph(after, size_of);
    int64_t rebuilt = 0;
    for (const Object* obj : after_stats.nodes) rebuilt += before.nodes.count(obj) == 0;
    EmitCounter(Policy::kName, fixture.name, "moved", "map_replace", "rebuilt", rebuilt);
    EmitAssert(Policy::kName, "rebuilt/" + fixture.name + "/moved/map_replace",
               fixture.expect_rebuilt_moved < 0 || rebuilt == fixture.expect_rebuilt_moved,
               "measured=" + std::to_string(rebuilt) +
                   " declared=" + std::to_string(fixture.expect_rebuilt_moved));
  }

  // ---- dispatch-count equality ---------------------------------------------
  //
  // A "dispatch" is one entry into a registered structural hook, counted by wrapping every
  // hook in the type-attribute columns for an untimed pass.  Arms are compared inside the
  // family that claims the same traversal:
  //
  //   * every walk arm, and every map arm whose callback never matches, must enter one hook
  //     per node occurrence in the fixture;
  //   * a map arm whose callback does match trades a hook dispatch for a callback on each
  //     first occurrence of a matched identity, and the identity remap serves every later
  //     occurrence, so its count is `occurrences - remap hits` -- a declared fixture
  //     property, not a number read off the run.
  {
    Any probe = fixture.build();
    int64_t floor_walk = CountDispatches([&] { Policy::RunWalkArm(ArmId::kWalkFloor, probe); });
    int64_t walk = CountDispatches([&] { Policy::RunWalkArm(ArmId::kWalk, probe); });
    int64_t walk_never = CountDispatches([&] { Policy::RunWalkArm(ArmId::kWalkNever, probe); });
    int64_t walk_old = CountDispatches([&] { Policy::RunWalkArm(ArmId::kWalkOld, probe); });
    EmitCounter(Policy::kName, fixture.name, "-", "walk_floor", "dispatches", floor_walk);
    EmitCounter(Policy::kName, fixture.name, "-", "walk", "dispatches", walk);
    EmitCounter(Policy::kName, fixture.name, "-", "walk_never", "dispatches", walk_never);
    EmitCounter(Policy::kName, fixture.name, "-", "walk_old", "dispatches", walk_old);
    EmitAssert(Policy::kName, "dispatch/" + fixture.name + "/walk-arms-agree",
               !CanCountDispatches() ||
                   (floor_walk == stats.occurrences && walk == stats.occurrences &&
                    walk_never == stats.occurrences &&
                    (!Policy::kOldGoesThroughEngine || walk_old == stats.occurrences)),
               "occurrences=" + std::to_string(stats.occurrences) +
                   " walk_floor=" + std::to_string(floor_walk) + " walk=" + std::to_string(walk) +
                   " walk_never=" + std::to_string(walk_never) +
                   " walk_old=" + std::to_string(walk_old));

    Any slot = fixture.build();
    int64_t floor_map =
        CountDispatches([&] { Policy::RunMapArm(ArmId::kMapFloor, Ownership::kRetained, &slot); });
    int64_t map_never =
        CountDispatches([&] { Policy::RunMapArm(ArmId::kMapNever, Ownership::kRetained, &slot); });
    int64_t map_identity =
        CountDispatches([&] { Policy::RunMapArm(ArmId::kMapIdentity, Ownership::kRetained, &slot); });
    int64_t map_replace =
        CountDispatches([&] { Policy::RunMapArm(ArmId::kMapReplace, Ownership::kRetained, &slot); });
    int64_t map_old =
        CountDispatches([&] { Policy::RunMapArm(ArmId::kMapOld, Ownership::kRetained, &slot); });
    EmitCounter(Policy::kName, fixture.name, "retained", "map_floor", "dispatches", floor_map);
    EmitCounter(Policy::kName, fixture.name, "retained", "map_never", "dispatches", map_never);
    EmitCounter(Policy::kName, fixture.name, "retained", "map_identity", "dispatches", map_identity);
    EmitCounter(Policy::kName, fixture.name, "retained", "map_replace", "dispatches", map_replace);
    EmitCounter(Policy::kName, fixture.name, "retained", "map_old", "dispatches", map_old);
    EmitAssert(Policy::kName, "dispatch/" + fixture.name + "/unmatched-map-arms-agree",
               !CanCountDispatches() ||
                   (floor_map == stats.occurrences && map_never == stats.occurrences),
               "occurrences=" + std::to_string(stats.occurrences) +
                   " map_floor=" + std::to_string(floor_map) +
                   " map_never=" + std::to_string(map_never));
    EmitAssert(Policy::kName, "dispatch/" + fixture.name + "/walk-and-map-agree",
               !CanCountDispatches() || floor_walk == floor_map,
               "walk_floor=" + std::to_string(floor_walk) +
                   " map_floor=" + std::to_string(floor_map));
    EmitAssert(Policy::kName, "dispatch/" + fixture.name + "/matched-map-arms-agree",
               !CanCountDispatches() ||
                   (map_identity == map_replace &&
                    (!Policy::kOldGoesThroughEngine || map_replace == map_old)),
               "map_identity=" + std::to_string(map_identity) +
                   " map_replace=" + std::to_string(map_replace) +
                   " map_old=" + std::to_string(map_old));
    if (fixture.expect_remap_hits >= 0 && CanCountDispatches()) {
      EmitAssert(Policy::kName, "dispatch/" + fixture.name + "/matched-map-remap-hits",
                 map_identity == stats.occurrences - fixture.expect_remap_hits,
                 "map_identity=" + std::to_string(map_identity) + " occurrences=" +
                     std::to_string(stats.occurrences) +
                     " declared_remap_hits=" + std::to_string(fixture.expect_remap_hits));
    }
  }

  // ---- timing ---------------------------------------------------------------
  const int64_t n = stats.unique_nodes;
  const int kSmallBatch = 64;
  std::unordered_map<std::string, double> timings;

  for (const char* ownership : {"retained", "moved"}) {
    EmitFixture(Policy::kName, fixture.name, ownership, n, stats.occurrences, stats.working_set, fits,
                fixture.expect_changed, n - fixture.expect_changed);
  }

  std::vector<ArmId> walk_arms = {ArmId::kWalkFloor, ArmId::kWalk, ArmId::kWalkNever,
                                  ArmId::kWalkOld};
  for (ArmId id : walk_arms) {
    Any pinned = fixture.build();
    ArmBinding binding;
    binding.id = id;
    binding.run = [&, id](int) { Policy::RunWalkArm(id, pinned); };
    double ns = MeasureNsPerTraversal(binding);
    timings[std::string(ArmSpecOf(id).name)] = ns;
    EmitResult(Policy::kName, fixture.name, "-", ArmSpecOf(id).name, n, ns);
  }

  std::vector<ArmId> map_arms = {ArmId::kMapFloor, ArmId::kMapNever, ArmId::kMapIdentity,
                                 ArmId::kMapReplace, ArmId::kMapOld};
  for (Ownership ownership : {Ownership::kRetained, Ownership::kMoved}) {
    for (ArmId id : map_arms) {
      const bool rebuilds = ArmSpecOf(id).rebuilds;
      // An in-place rebuilding arm on a fixture with a pointer-shared subtree un-shares it,
      // so the fixture is not stationary and must be rebuilt: a pool of independent copies
      // keeps the timed batch free of construction while the shape stays what it declares.
      const bool needs_pool =
          ownership == Ownership::kMoved && rebuilds && fixture.has_sharing;
      std::vector<Any> pool;
      ArmBinding binding;
      binding.id = id;
      binding.ownership = ownership;
      if (needs_pool) {
        binding.batch = kSmallBatch;
        binding.prepare = [&] {
          pool.clear();
          for (int i = 0; i < kSmallBatch; ++i) pool.push_back(fixture.build());
        };
        binding.run = [&, id, ownership](int i) { Policy::RunMapArm(id, ownership, &pool[i]); };
      } else {
        pool.push_back(fixture.build());
        binding.run = [&, id, ownership](int) { Policy::RunMapArm(id, ownership, &pool[0]); };
      }
      double ns = MeasureNsPerTraversal(binding);
      timings[std::string(OwnershipName(ownership)) + "/" + ArmSpecOf(id).name] = ns;
      EmitResult(Policy::kName, fixture.name, OwnershipName(ownership), ArmSpecOf(id).name, n, ns);
    }
  }

  // ---- walk must be faster than map on the same fixture ---------------------
  //
  // A walk returns a status word per node; a map returns an owned value, touches a refcount
  // and has its parent run `same_as`.  A map can therefore never be cheaper than the walk it
  // dispatches identically to.  #368 published a table where it was, and the contradiction
  // was caught in review rather than by the tool. Here it fails the run.
  for (const char* ownership : {"retained", "moved"}) {
    EmitAssert(Policy::kName, std::string("walk-faster-than-map/") + fixture.name + "/" +
                                  ownership + "/floor",
               timings["walk_floor"] < timings[std::string(ownership) + "/map_floor"],
               "walk_floor=" + std::to_string(timings["walk_floor"]) + " map_floor=" +
                   std::to_string(timings[std::string(ownership) + "/map_floor"]));
    EmitAssert(Policy::kName, std::string("walk-faster-than-map/") + fixture.name + "/" +
                                  ownership + "/hooked",
               timings["walk"] < timings[std::string(ownership) + "/map_identity"],
               "walk=" + std::to_string(timings["walk"]) + " map_identity=" +
                   std::to_string(timings[std::string(ownership) + "/map_identity"]));
  }
}

}  // namespace bench
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_BENCHMARKS_CPP_STRUCTURAL_BENCH_COMMON_H_
