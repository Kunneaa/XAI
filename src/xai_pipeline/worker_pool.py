"""Timeout-protected persistent SymPy worker pool."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class WorkerPoolResult:
    ok: bool
    value: object
    issues: list[str]
    trace: dict

    def to_dict(self) -> dict:
        return {"ok": self.ok, "value": self.value, "issues": list(self.issues), "trace": dict(self.trace)}


class SympyWorkerPool:
    """Single-worker warm pool with timeout and replacement semantics.

    SymPy is loaded inside a long-lived child process and jobs are sent through
    queues. If a job exceeds its timeout, the child process is killed and a new
    worker is started for the next request. This keeps symbolic execution warm
    while preserving the core fail-closed timeout boundary.
    """

    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout_seconds = timeout_seconds
        self.jobs_submitted = 0
        self.workers_replaced = 0
        self._job_id = 0
        self._lock = Lock()
        self._ctx = _mp_context()
        self._request_queue = None
        self._response_queue = None
        self._process = None

    def solve(self, *, equations: list[str], targets: list[str], timeout_seconds: float | None = None) -> WorkerPoolResult:
        with self._lock:
            if not equations or not targets:
                return WorkerPoolResult(False, None, ["missing_equations_or_targets"], _trace(equations, targets, timeout_seconds or self.timeout_seconds))
            self._ensure_worker()
            self.jobs_submitted += 1
            self._job_id += 1
            job_id = self._job_id
            timeout = timeout_seconds or self.timeout_seconds
            assert self._request_queue is not None
            assert self._response_queue is not None
            self._request_queue.put({"job_id": job_id, "equations": list(equations), "targets": list(targets)})
            started = time.monotonic()
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    self._replace_worker()
                    trace = self._pool_trace(equations, targets, timeout)
                    trace["timed_out_job_id"] = job_id
                    return WorkerPoolResult(False, None, ["sympy_timeout"], trace)
                try:
                    payload = self._response_queue.get(timeout=remaining)
                except queue.Empty:
                    self._replace_worker()
                    trace = self._pool_trace(equations, targets, timeout)
                    trace["timed_out_job_id"] = job_id
                    return WorkerPoolResult(False, None, ["sympy_timeout"], trace)
                if payload.get("job_id") != job_id:
                    continue
                trace = self._pool_trace(equations, targets, timeout)
                trace.update(payload.get("trace") or {})
                if not payload.get("ok"):
                    return WorkerPoolResult(False, None, payload.get("issues", ["sympy_worker_failed"]), trace)
                return WorkerPoolResult(True, payload.get("value"), [], trace)

    def close(self) -> None:
        with self._lock:
            self._stop_worker()

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(target=_persistent_sympy_worker, args=(self._request_queue, self._response_queue), daemon=True)
        self._process.start()

    def _stop_worker(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if self._request_queue is not None:
                self._request_queue.put(None)
            process.join(0.2)
            if process.is_alive():
                process.terminate()
                process.join(0.2)
        finally:
            self._process = None
            self._request_queue = None
            self._response_queue = None

    def _replace_worker(self) -> None:
        self._stop_worker()
        self.workers_replaced += 1
        self._ensure_worker()

    def _pool_trace(self, equations: list[str], targets: list[str], timeout_seconds: float) -> dict:
        trace = _trace(equations, targets, timeout_seconds)
        trace.update(
            {
                "pool": "persistent_warm_worker",
                "jobs_submitted": self.jobs_submitted,
                "workers_replaced": self.workers_replaced,
                "worker_pid": self._process.pid if self._process is not None else None,
            }
        )
        return trace


_DEFAULT_POOL: SympyWorkerPool | None = None


def get_default_sympy_pool() -> SympyWorkerPool:
    global _DEFAULT_POOL
    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = SympyWorkerPool()
        atexit.register(_DEFAULT_POOL.close)
    return _DEFAULT_POOL


def solve_symbolic_supervised(*, equations: list[str], targets: list[str], timeout_seconds: float = 3.0) -> WorkerPoolResult:
    """One-shot supervised solve kept for tests and isolated calls."""

    if not equations or not targets:
        return WorkerPoolResult(False, None, ["missing_equations_or_targets"], _trace(equations, targets, timeout_seconds))
    ctx = _mp_context()
    queue = ctx.Queue()
    process = ctx.Process(target=_sympy_worker, args=(equations, targets, queue), daemon=True)
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(0.2)
        return WorkerPoolResult(False, None, ["sympy_timeout"], _trace(equations, targets, timeout_seconds))
    if queue.empty():
        return WorkerPoolResult(False, None, ["sympy_worker_no_result"], _trace(equations, targets, timeout_seconds))
    payload = queue.get()
    if not payload.get("ok"):
        return WorkerPoolResult(False, None, payload.get("issues", ["sympy_worker_failed"]), _trace(equations, targets, timeout_seconds))
    return WorkerPoolResult(True, payload.get("value"), [], _trace(equations, targets, timeout_seconds))


def _mp_context():
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return mp.get_context("fork")
    return mp.get_context()


def _trace(equations: list[str], targets: list[str], timeout_seconds: float) -> dict:
    return {
        "stage": "sympy_worker_pool",
        "equation_count": len(equations),
        "target_count": len(targets),
        "timeout_seconds": timeout_seconds,
    }


def _persistent_sympy_worker(request_queue, response_queue) -> None:
    try:
        import sympy as sp
    except Exception as exc:
        while True:
            job = request_queue.get()
            if job is None:
                return
            response_queue.put({"job_id": job.get("job_id"), "ok": False, "issues": [f"sympy_not_available:{type(exc).__name__}"], "trace": {"worker": "persistent_sympy_worker"}})
    while True:
        job = request_queue.get()
        if job is None:
            return
        payload = _solve_sympy_payload(job.get("equations") or [], job.get("targets") or [], sp)
        payload["job_id"] = job.get("job_id")
        payload.setdefault("trace", {})["worker"] = "persistent_sympy_worker"
        response_queue.put(payload)


def _sympy_worker(equations: list[str], targets: list[str], queue) -> None:
    try:
        import sympy as sp
    except Exception as exc:
        queue.put({"ok": False, "issues": [f"sympy_not_available:{type(exc).__name__}"]})
        return
    queue.put(_solve_sympy_payload(equations, targets, sp))


def _solve_sympy_payload(equations: list[str], targets: list[str], sp) -> dict:
    try:
        symbols = {name: sp.Symbol(name, real=True) for name in _symbol_names(equations, targets)}
        parsed_equations = []
        for equation in equations:
            equation = str(equation).replace("^", "**")
            if "=" in equation:
                lhs, rhs = equation.split("=", 1)
                parsed_equations.append(sp.Eq(sp.sympify(lhs, locals=symbols), sp.sympify(rhs, locals=symbols)))
            else:
                parsed_equations.append(sp.sympify(equation, locals=symbols))
        target_symbols = [symbols[target] for target in targets if target in symbols]
        if not target_symbols:
            return {"ok": False, "issues": ["target_symbols_not_found"], "trace": {"symbol_count": len(symbols)}}
        solutions = sp.solve(parsed_equations, target_symbols, dict=True)
        serializable = [{str(key): float(value) if value.is_number else str(value) for key, value in solution.items()} for solution in solutions]
        return {"ok": True, "value": serializable, "trace": {"symbol_count": len(symbols), "solution_count": len(serializable)}}
    except Exception as exc:
        return {"ok": False, "issues": [f"sympy_error:{type(exc).__name__}"], "trace": {"error": str(exc)[:300]}}


def _symbol_names(equations: list[str], targets: list[str]) -> set[str]:
    import re

    names = set(targets)
    for equation in equations:
        names.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", equation))
    names.difference_update({"sqrt", "sin", "cos", "tan", "pi", "E"})
    return names
