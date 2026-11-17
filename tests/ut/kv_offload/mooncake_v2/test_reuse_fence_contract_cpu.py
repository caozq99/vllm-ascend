"""CPU-only contract tests for Store-to-Push reuse fences.

Run this file directly: no project imports, pytest conftest, or device probes.
The tested method bodies and reuse_sources helper come from the working tree.
"""
import ast
import concurrent.futures
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
STORE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise"
PUSH = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/push_worker.py"


def extract(path, name, namespace=None, class_name=None):
    tree = ast.parse(path.read_text())
    scope = tree
    if class_name:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    candidates = [n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(candidates) != 1:
        raise AssertionError((path, name, len(candidates)))
    function = candidates[0]
    function.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    ns = {} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns[name]


reuse_sources = extract(STORE / "layerwise_cache_layout.py", "reuse_sources")
store_wait = extract(STORE / "kv_transfer.py", "_wait_for_reuse_saves",
                     class_name="KVCacheStoreLayerRecvingThread")


def push_wait(wait=concurrent.futures.wait):
    return extract(PUSH, "wait_for_layer_reuse", {"reuse_sources": reuse_sources, "wait_futures": wait})


def completed(value=None, error=None):
    result = concurrent.futures.Future()
    if error is None:
        result.set_result(value)
    else:
        result.set_exception(error)
    return result


def producer(writes):
    return SimpleNamespace(_prefilling_request_ids={"request"},
                           _slot_predecessors={19: (11, 15)}, _slot_writes=writes)


def store(external, events):
    return SimpleNamespace(
        _wait_for_event=lambda event, checker: events.append(("save", event)) or True,
        layer_save_finished_events={11: 11, 15: 15},
        save_failure_checker=lambda: events.append(("check",)),
        sync_save_events={i: SimpleNamespace(synchronize=lambda i=i: events.append(("sync", i)))
                          for i in (11, 15)},
        external_slot_release_waiter=external,
    )


class ReuseFenceContractTest(unittest.TestCase):
    def test_store_passes_target_once_after_all_local_predecessors(self):
        events = []
        worker = store(lambda target: events.append(("external", target)), events)
        self.assertTrue(store_wait(worker, SimpleNamespace(layer_id=19, wait_for_save_layers=(11, 15))))
        self.assertEqual(events, [("save", 11), ("check",), ("sync", 11),
                                  ("save", 15), ("check",), ("sync", 15), ("external", 19)])

    def test_external_target_fence_is_not_skipped_without_local_predecessors(self):
        events = []
        worker = store(lambda target: events.append(("external", target)), events)
        self.assertTrue(store_wait(worker, SimpleNamespace(layer_id=19, wait_for_save_layers=())))
        self.assertEqual(events, [("external", 19)])

    def test_push_waits_for_second_predecessor(self):
        second = concurrent.futures.Future()
        waiting = threading.Event()
        finished = threading.Event()
        errors = []

        def observed_wait(writes):
            if second in writes:
                waiting.set()
            return concurrent.futures.wait(writes)

        worker = producer({11: (completed(),), 15: (second,)})

        def run():
            try:
                push_wait(observed_wait)(worker, 19)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(waiting.wait(2), "Second predecessor was not reached")
            self.assertFalse(finished.is_set(), "Fence released before second WRITE completed")
        finally:
            second.set_result(None)
            thread.join(2)
        self.assertTrue(finished.is_set())
        self.assertEqual(errors, [])

    def test_push_raises_returned_write_error(self):
        worker = producer({11: (completed(),), 15: (completed("WRITE failed: ret=-1"),)})
        with self.assertRaisesRegex(RuntimeError, "WRITE failed"):
            push_wait()(worker, 19)

    def test_push_propagates_future_exception(self):
        worker = producer({11: (completed(),), 15: (completed(error=ValueError("transport exception")),)})
        with self.assertRaisesRegex(ValueError, "transport exception"):
            push_wait()(worker, 19)

    def test_push_rejects_missing_save_hook(self):
        with self.assertRaisesRegex(RuntimeError, "missing.*15"):
            push_wait()(producer({11: (completed(),)}), 19)

    def test_inactive_push_needs_no_fence(self):
        worker = producer({})
        worker._prefilling_request_ids.clear()
        push_wait()(worker, 19)

    def test_store_push_chain_surfaces_second_predecessor_failure(self):
        # 11 and 15 have no predecessors themselves: passing either as a target
        # would silently skip the actual target 19's pending/failed WRITEs.
        push = producer({11: (completed(),), 15: (completed("second WRITE failed"),)})
        events = []
        receiver = store(lambda target: push_wait()(push, target), events)
        with self.assertRaisesRegex(RuntimeError, "second WRITE failed"):
            store_wait(receiver, SimpleNamespace(layer_id=19, wait_for_save_layers=(11, 15)))


    def test_range_caller_invokes_external_target_once(self):
        events = []
        worker = store(lambda target: events.append(("external", target)), events)
        worker._wait_for_reuse_saves = lambda data: store_wait(worker, data)
        worker.layer_load_finished_events = {19: threading.Event()}
        worker._group_load_indices = {}
        worker.request_queue = SimpleNamespace(task_done=lambda: None)
        worker.get_event = threading.Event()
        method = extract(STORE / "kv_transfer.py", "_handle_range_layer_task",
                         class_name="KVCacheStoreLayerRecvingThread")
        method(worker, SimpleNamespace(layer_id=19, wait_for_save_layers=(),
                                       attention_start_gate=None, transfer_tasks=[]))
        self.assertEqual(events, [("external", 19)])
        self.assertTrue(worker.layer_load_finished_events[19].is_set())

    def test_gva_disabled_load_caller_invokes_external_target_once(self):
        events = []
        worker = store(lambda target: events.append(("external", target)), events)
        worker._wait_for_reuse_saves = lambda data: store_wait(worker, data)
        worker.load_disabled = SimpleNamespace(is_set=lambda: True)
        method = extract(STORE / "kv_transfer.py", "_copy_gva_layers",
                         class_name="KVCacheStoreLayerRecvingThread")
        self.assertFalse(method(worker, SimpleNamespace(layer_id=19, wait_for_save_layers=(),
                                                        attention_start_gate=None, transfer_tasks=[])))
        self.assertEqual(events, [("external", 19)])

    def test_fenced_callers_do_not_repeat_external_wait(self):
        tree = ast.parse((STORE / "kv_transfer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "KVCacheStoreLayerRecvingThread")
        callers = []
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            calls = [n.func.attr for n in ast.walk(method)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            if "_wait_for_reuse_saves" in calls:
                callers.append(method.name)
                self.assertNotIn("external_slot_release_waiter", calls, method.name)
        self.assertEqual(set(callers), {"_handle_range_layer_task", "_copy_gva_layers"})

    def test_failed_local_save_wait_never_releases_to_external(self):
        events = []
        worker = store(lambda target: events.append(("external", target)), events)
        worker._wait_for_event = lambda event, checker: False
        self.assertFalse(store_wait(worker, SimpleNamespace(layer_id=19, wait_for_save_layers=(11, 15))))
        self.assertEqual(events, [])



    def test_shared_save_event_survives_two_target_readers(self):
        source = threading.Event()
        source.set()
        synced = []
        targets = []
        def bounded_wait(event, checker):
            # A consumed/cleared event must fail promptly, not hang the test.
            return event.wait(0.01)
        worker = SimpleNamespace(
            layer_save_finished_events={0: source},
            sync_save_events={0: SimpleNamespace(synchronize=lambda: synced.append(0))},
            save_failure_checker=None,
            _wait_for_event=bounded_wait,
            external_slot_release_waiter=targets.append,
        )
        for target in (1, 2):
            self.assertTrue(store_wait(worker, SimpleNamespace(
                layer_id=target, wait_for_save_layers=(0,))))
            self.assertTrue(source.is_set())
        self.assertEqual(synced, [0, 0])
        self.assertEqual(targets, [1, 2])

    def test_step_drain_rearms_save_event_only_after_all_readers(self):
        source = threading.Event()
        source.set()
        order = []
        def drain(**kwargs):
            self.assertTrue(source.is_set(), "Save completion cleared before reader drain")
            order.append(kwargs)
        worker = SimpleNamespace(
            layer_save_finished_events=[source],
            _drain_attention_transfers=drain,
            layerwise_offload=True,
            _check_layerwise_transfer_errors=lambda: order.append("check"),
            prefetch_layer_map={1: (0,), 2: (0,)},
        )
        method = extract(STORE / "pool_worker.py", "_wait_for_final_layer_save",
                         class_name="LayerwiseKVPoolWorker")
        method(worker, 1)
        self.assertEqual(order, [{"drain_recv": False, "full": True}, "check", {"full": True}])
        self.assertFalse(source.is_set())

if __name__ == "__main__":
    unittest.main(verbosity=2)
