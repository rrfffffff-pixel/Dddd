"""Benchmark runner for evaluating agent performance."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkTask:
    name: str
    description: str
    setup: str = ""
    expected_outcome: str = ""


@dataclass
class BenchmarkResult:
    task_name: str
    success: bool
    duration: float
    output: str = ""
    error: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    name: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def summary(self) -> str:
        if not self.results:
            return "No benchmark results"
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        durations = [r.duration for r in self.results]
        avg = statistics.mean(durations) if durations else 0
        fastest = min(durations) if durations else 0
        slowest = max(durations) if durations else 0
        lines = [
            f"Benchmark: {self.name}",
            f"Tasks: {passed}/{total} passed",
            f"Duration: avg={avg:.1f}s, fastest={fastest:.1f}s, slowest={slowest:.1f}s",
        ]
        for r in self.results:
            status = "PASS" if r.success else "FAIL"
            lines.append(f"  [{status}] {r.task_name} ({r.duration:.1f}s): {r.output[:100]}")
            if r.error:
                lines.append(f"         Error: {r.error[:100]}")
        return "\n".join(lines)


DEFAULT_BENCHMARKS = [
    BenchmarkTask(
        name="read_file",
        description="Read the main.py file from the project",
        expected_outcome="file contents",
    ),
    BenchmarkTask(
        name="list_files",
        description="List all Python files in the project",
        expected_outcome="Python files",
    ),
    BenchmarkTask(
        name="grep_search",
        description="Search for 'def ' in all Python files",
        expected_outcome="function definitions",
    ),
    BenchmarkTask(
        name="code_review",
        description="Review code quality of the agent.py file",
        expected_outcome="code review",
    ),
]


def run_benchmark(
    suite_name: str,
    tasks: list[BenchmarkTask],
    runner_fn: Callable[[BenchmarkTask], BenchmarkResult],
) -> BenchmarkSuite:
    suite = BenchmarkSuite(name=suite_name)
    for task in tasks:
        print(f"  Running: {task.name}... ", end="", flush=True)
        try:
            result = runner_fn(task)
        except Exception as e:
            result = BenchmarkResult(
                task_name=task.name, success=False, duration=0, error=str(e)
            )
        print(f"{'PASS' if result.success else 'FAIL'} ({result.duration:.1f}s)")
        suite.add(result)
    return suite
