"""Phase 8: the fitness signal.

Without this, "AURA got better" is unfalsifiable -- which is why the roadmap
put evaluation before learning, reversing the spec.
"""

from aura.eval.compare import Comparison, compare, load_report, save_report
from aura.eval.runner import SuiteRunner
from aura.eval.suite import SuiteError, load_suite
from aura.eval.types import Attempt, Category, Check, Report, Task, TaskResult

__all__ = [
    "Attempt",
    "Category",
    "Check",
    "Comparison",
    "Report",
    "SuiteError",
    "SuiteRunner",
    "Task",
    "TaskResult",
    "compare",
    "load_report",
    "load_suite",
    "save_report",
]
