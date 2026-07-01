"""DAG-based workflow engine for parallel and sequential task execution."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class StageType(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass
class StageNode:
    id: str
    name: str
    stage_type: StageType
    tasks: list[WorkflowTask] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, done, failed
    task_timeout: float | None = None  # per-task timeout in seconds (None = no timeout)


@dataclass
class WorkflowTask:
    id: str
    description: str
    agent_name: str
    status: str = "pending"
    result: str = ""
    error: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class StageResult:
    stage_id: str
    success: bool
    task_results: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    total_tasks: int = 0
    succeeded_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    duration_seconds: float = 0.0


class WorkflowEngine:
    """DAG-based workflow engine supporting parallel and sequential stages."""

    def __init__(
        self,
        agents: dict[str, Any],
        max_workers: int = 4,
        progress_callback: Callable[[str, str, str], None] | None = None,
        default_task_timeout: float | None = None,
    ) -> None:
        self.agents = agents
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.default_task_timeout = default_task_timeout
        self._cancelled = threading.Event()
        self._cancel_lock = threading.Lock()

    def _notify_progress(self, stage_id: str, status: str, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(stage_id, status, message)

    def cancel(self) -> None:
        """Signal the workflow to abort after the current task completes."""
        with self._cancel_lock:
            self._cancelled.set()
        logger.info("Workflow cancel requested")

    @property
    def is_cancelled(self) -> bool:
        with self._cancel_lock:
            return self._cancelled.is_set()

    def _execute_task(self, task: WorkflowTask, stage_id: str) -> str:
        agent = self.agents.get(task.agent_name)
        if agent is None:
            task.status = "failed"
            task.error = f"Agent '{task.agent_name}' not available"
            task.end_time = time.monotonic()
            task.duration_seconds = task.end_time - task.start_time
            self._notify_progress(stage_id, "failed", task.error)
            return ""

        task.status = "running"
        task.start_time = time.monotonic()
        self._notify_progress(stage_id, "running", task.description)

        try:
            result = agent.run(task.description)
        except Exception as e:
            task.status = "failed"
            task.error = f"Agent raised exception: {e}"
            task.end_time = time.monotonic()
            task.duration_seconds = task.end_time - task.start_time
            self._notify_progress(stage_id, "failed", task.error)
            return ""
        finally:
            task.end_time = time.monotonic()
            task.duration_seconds = task.end_time - task.start_time

        task.result = result

        if self._is_failure(result):
            task.status = "failed"
            task.error = result
            self._notify_progress(stage_id, "failed", result)
            return ""

        task.status = "done"
        self._notify_progress(stage_id, "done", result)
        return result

    def _execute_task_with_timeout(
        self, task: WorkflowTask, stage_id: str, timeout: float | None, executor: ThreadPoolExecutor
    ) -> str:
        if timeout is None:
            return self._execute_task(task, stage_id)

        future = executor.submit(self._execute_task, task, stage_id)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            task.status = "failed"
            task.error = f"Task timed out after {timeout}s"
            task.end_time = time.monotonic()
            task.duration_seconds = task.end_time - task.start_time
            self._notify_progress(stage_id, "failed", task.error)
            future.cancel()
            return ""
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.end_time = time.monotonic()
            task.duration_seconds = task.end_time - task.start_time
            self._notify_progress(stage_id, "failed", str(e))
            return ""

    def _is_failure(self, result: str) -> bool:
        indicators = ["error", "failed", "exception", "traceback"]
        lower = result.lower()
        return any(ind in lower for ind in indicators)

    def validate_dependencies(self, stages: dict[str, StageNode]) -> list[str]:
        """Validate stage dependency graph. Returns list of error messages (empty = valid)."""
        errors: list[str] = []
        stage_ids = set(stages.keys())

        # Check for missing dependencies
        for sid, stage in stages.items():
            for dep in stage.depends_on:
                if dep not in stage_ids:
                    errors.append(f"Stage '{sid}' depends on unknown stage '{dep}'")

        # Cycle detection via topological sort (Kahn's algorithm)
        if errors:
            return errors

        in_degree: dict[str, int] = {sid: 0 for sid in stages}
        adj: dict[str, list[str]] = {sid: [] for sid in stages}
        for sid, stage in stages.items():
            for dep in stage.depends_on:
                adj[dep].append(sid)
                in_degree[sid] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(stages):
            cycle_stages = [sid for sid, deg in in_degree.items() if deg > 0]
            errors.append(f"Dependency cycle detected among stages: {', '.join(cycle_stages)}")

        return errors

    def _get_ready_stages(self, stages: dict[str, StageNode]) -> list[StageNode]:
        done_ids = {sid for sid, s in stages.items() if s.status == "done"}
        return [
            s for s in stages.values()
            if s.status == "pending" and all(d in done_ids for d in s.depends_on)
        ]

    def _execute_stage(self, stage: StageNode) -> StageResult:
        stage.status = "running"
        self._notify_progress(stage.id, "running", stage.name)
        stage_start = time.monotonic()

        task_results: dict[str, str] = {}
        errors: dict[str, str] = {}
        results_lock = threading.Lock()
        skipped: int = 0

        task_timeout = stage.task_timeout or self.default_task_timeout

        def run_task(task: WorkflowTask) -> str:
            if self.is_cancelled:
                task.status = "failed"
                task.error = "Workflow cancelled"
                return ""
            result = self._execute_task(task, stage.id)
            with results_lock:
                if result:
                    task_results[task.id] = result
                if task.error:
                    errors[task.id] = task.error
            return result

        if stage.stage_type == StageType.SEQUENTIAL:
            for task in stage.tasks:
                if self.is_cancelled:
                    skipped += len(stage.tasks) - stage.tasks.index(task)
                    for t in stage.tasks[stage.tasks.index(task):]:
                        t.status = "failed"
                        t.error = "Workflow cancelled"
                    break
                run_task(task)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, max(len(stage.tasks), 1))) as executor:
                futures: dict[Future, WorkflowTask] = {}
                for task in stage.tasks:
                    if self.is_cancelled:
                        task.status = "failed"
                        task.error = "Workflow cancelled"
                        skipped += 1
                        continue
                    future = executor.submit(run_task, task)
                    futures[future] = task

                for future in futures:
                    try:
                        future.result(timeout=task_timeout)
                    except FuturesTimeoutError:
                        task = futures[future]
                        task.status = "failed"
                        task.error = f"Stage task timed out after {task_timeout}s"
                        task.end_time = time.monotonic()
                        task.duration_seconds = task.end_time - task.start_time
                        with results_lock:
                            errors[task.id] = task.error
                        self._notify_progress(stage.id, "failed", task.error)
                        future.cancel()
                    except Exception as e:
                        task = futures[future]
                        task.status = "failed"
                        task.error = str(e)
                        with results_lock:
                            errors[task.id] = str(e)
                        self._notify_progress(stage.id, "failed", str(e))

        stage_end = time.monotonic()
        total = len(stage.tasks)
        failed = len(errors)
        succeeded = total - failed - skipped
        success = failed == 0 and skipped == 0
        stage.status = "done" if success else "failed"
        return StageResult(
            stage_id=stage.id,
            success=success,
            task_results=task_results,
            errors=errors,
            total_tasks=total,
            succeeded_tasks=succeeded,
            failed_tasks=failed,
            skipped_tasks=skipped,
            duration_seconds=round(stage_end - stage_start, 3),
        )

    def run(self, stages: dict[str, StageNode], max_rounds: int = 50) -> list[StageResult]:
        """Execute stages respecting dependency graph."""
        with self._cancel_lock:
            self._cancelled.clear()

        # Validate dependencies before execution
        dep_errors = self.validate_dependencies(stages)
        if dep_errors:
            logger.error("Dependency validation failed: %s", "; ".join(dep_errors))
            return []

        results: list[StageResult] = []

        for _ in range(max_rounds):
            if self.is_cancelled:
                logger.info("Workflow cancelled, stopping execution")
                break

            ready = self._get_ready_stages(stages)
            if not ready:
                break

            for stage in ready:
                if self.is_cancelled:
                    break
                result = self._execute_stage(stage)
                results.append(result)

                # Skip downstream stages if this one failed and has no tolerance
                # (fail-fast: stop if any stage fails)
                if not result.success:
                    logger.warning(
                        "Stage '%s' failed (%d/%d tasks), skipping downstream",
                        stage.id, result.failed_tasks, result.total_tasks,
                    )

        return results
