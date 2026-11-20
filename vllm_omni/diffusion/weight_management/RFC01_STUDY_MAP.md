# RFC 01 Study Map — Startup Transaction, Selector, Coexistence

> **What this is:** a local *study* implementation of child RFC 01 (issue
> [#9874](https://github.com/vllm-project/vllm-omni/issues/9874), umbrella
> [#9536](https://github.com/vllm-project/vllm-omni/issues/9536)), built to be
> read section-by-section against the RFC. **Not for upstream submission** —
> 01-1 and 01-4 already exist as maintainer PRs (#9725, #9726). This is for
> understanding how the unified weight management starts up.
>
> **How to study:** open the RFC issue and this folder side by side. For each
> PR below, read the RFC sections listed, then read the file:symbol, then read
> the test that pins its behavior. The code comments repeat the `RFC 01 §x`
> anchors so you never lose your place.
>
> **Running the tests:** the normal `pytest` can't collect in this checkout
> (the installed `vllm` wheel mismatches `vllm_omni`). The code is verified with
> an isolated runner that bypasses the broken package `__init__`. Once the env
> is fixed (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`), the test files under
> `tests/diffusion/weight_management/` run directly.

## The seven PRs

| PR | RFC §§ | File(s) → key symbols | Status | What to study |
|----|--------|-----------------------|--------|---------------|
| **01-1** | §2, §3, §5.2 | `interfaces.py` → `LoadState`, `Topology`, `StartupContext`, 6 Protocols · `errors.py` → `WeightManagementError` + subclasses · `startup.py` → `ResourceKind`, `ResourceLedger` | ✅ done | The **contract** (Protocols + data types), the **error taxonomy**, and the **ordered exactly-once cleanup** (`ResourceLedger`). |
| **01-2** | §1 | `startup.py` → `WeightManagement`, `parse_weight_management`, `is_unified`, `set/clear/process_weight_management`, `check_unified_offload_conflicts` | ✅ logic (CLI/config propagation across the 5 files in §1.1 deferred) | How ONE flag picks legacy vs unified, and the **two readers** (config-level vs process-level) and owner-refcounted process state. |
| **01-3** | §9.2 | `interfaces.py` → `ExecutionOffloadFacts` · `startup.py` → `execution_offload_facts` | ✅ core (§9.1 offloader/config rewrite + engine/executor migration deferred) | The **coexistence pattern**: one function answers offload questions so call sites don't branch on the selector. Function-local imports (enforced by T1). |
| **01-4** | §7, §6.2 (topology) | `group.py` → `ReplicaGroup.exchange`, `RoundMessage`, `Point0/1Payload`, `assemble_topology`, `create_allgather_groups`, `read_rank_coordinates` | ✅ done (world-size-1 path; real `new_group` function-local) | The **cross-rank round protocol**: barrier + all-gather + "any rank fails ⇒ all fail." Topology assembly. AllGather group partitioning. |
| **01-5** | §4, §6, §5.1 | `startup.py` → `run_unified_startup`, `_Startup`, `WeightManagementHandle`, `EntryPointFactory`, `_wrap`, `_close_owner` | ✅ done (test doubles) | **The payoff.** The 13-step lifecycle, `LoadState` transitions, the ownership chain Election→Storage→Transaction→Runtime, failure release in ledger order. |
| **01-6** | §8, §10 | `dispatch.py` → `load_model_unified` (11-step §8.1), `close_weight_management`, `worker_shutdown_unified`, `SleepMemoryPools`, `sleep/wake_unified` · `startup.py` → `FaultPlan` (§10) | ✅ study module (real worker integration deferred) | Where the selector plugs into the worker: the 11-step unified load, shutdown order, sleep/wake, and fault injection for testing failures at any point. |
| **01-7** | §11, §9.4 | `tests/.../coexistence/registry.py` → `CoexistenceRow`, `ROWS`, 4 allowlists · `test_static_dependencies.py` → T1, T8 | ✅ core (T1+T8 runnable; T2-T7 described) | The **machine-enforced** layering: T1 AST-scans import direction (actually run), T8 SHA256-freezes the allowlists. Read §11. |

## RFC section → where it lives

| RFC 01 § | Topic | Code |
|----------|-------|------|
| §1.1 | selector values + propagation | *(deferred — 5 config files)* |
| §1.2 | selector functions, import discipline | `startup.py` selector block |
| §1.3 | unified-mode offload rejection | `startup.py` `check_unified_offload_conflicts` |
| §2 | interfaces (Protocols + data types) | `interfaces.py` |
| §3 | error types + message format | `errors.py` |
| §4 | state machine (`LoadState`) | `interfaces.py` `LoadState` + `startup.py` `_Startup.enter` |
| §5.1 | ownership chain | `startup.py` `_close_owner`, `_Startup.owner` |
| §5.2 | resource ledger | `startup.py` `ResourceLedger` |
| §6.1 | entry point, factory, handle | `startup.py` `run_unified_startup`, `EntryPointFactory`, `WeightManagementHandle` |
| §6.2 | the 13 steps | `startup.py` `_Startup` step methods |
| §7.1 | round protocol | `group.py` `ReplicaGroup.exchange`, `RoundMessage`, `raise_if_any_failed` |
| §7.2 | confirmation points | `group.py` payloads + `assemble_topology`, `create_allgather_groups` |
| §8 | worker/runner dispatch | `dispatch.py` |
| §9.2 | execution facts | `startup.py` `execution_offload_facts` |
| §9.4 | coexistence registry | `tests/.../coexistence/registry.py` |
| §10 | fault injection | `startup.py` `FaultPlan`, `InjectedFault` |
| §11 | static dependency tests | `tests/.../test_static_dependencies.py` (T1, T8) |

## The lifecycle at a glance (RFC 01 §6.2)

`run_unified_startup` runs these in order; **confirm** steps are the cross-rank
points where the whole replica must agree. Ownership moves at prepare/commit.

```
 1 construct_modules   factory.create()                        (StackUnavailableError if absent)
 2 build_structure     loader.build_structure()                -> STRUCTURE_READY
 · confirm_topology    POINT 0: gather records -> Topology
 4 plan                offloader.plan()  (pure)                -> PLANNED
 5 preflight           loader.preflight() + hwm.layout()       -> SOURCES_PREPARED
 · confirm_admission   POINT 1 (2 rounds): fingerprints -> groups -> elect -> admit -> ADMITTED
 7 allocate            hwm.allocate()            owner = HostStorage
 8 prepare             offloader.prepare_...()   owner = LoadTransaction
 · confirm_ready       POINT 2                                 -> TRANSACTION_READY
   connect_writer      storage.connect_writer()
 9 write               loader.write()                          -> WEIGHTS_READY
10 finalize            loader.finalize(); loader.close()        -> FINALIZED
11 commit              transaction.commit()      owner = OffloadRuntime -> COMMITTED
 · confirm_commit      POINT 3
12 compile_setup       offloader.setup_compile() (best-effort)
13 (runner)            handle.mark_in_service()                -> IN_SERVICE
```

Any failure → `_Startup.release()` closes the current `owner` (abort transaction
/ close runtime/storage/election), closes the Loader, discards the model →
`FAILED`. Other ranks learn at the next confirmation point and raise
`PeerFailureError`.

## Tests (behavior pinned per PR)

| Test file | Pins |
|-----------|------|
| `test_resource_ledger.py` | 01-1 ledger ordering / idempotency / transfer / exception aggregation |
| `test_selector.py` | 01-2 parse, is_unified, process-state refcount + conflict, offload gate |
| `test_execution_facts.py` | 01-3 unified all-false + legacy query mapping |
| `test_startup_driver.py` | 01-5 happy path → COMMITTED, handle close, failure release (incl. transaction abort), fault injection |
| `test_dispatch.py` | 01-6 handle wiring, one-startup guard, idempotent close, shutdown clears state, sleep/wake |
| `test_static_dependencies.py` | 01-7 T1 import direction (scanned), T8 frozen allowlists |

**Total: 39 tests passing** (via the isolated runner; see the note at the top).
