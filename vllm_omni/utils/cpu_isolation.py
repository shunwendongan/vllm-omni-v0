"""Host-thread CPU isolation for RTF load stability.

Binds hot host threads (orchestrator, stage engine cores) to dedicated CPU
cores and raises their scheduling priority so Python dispatch overhead stays
low even when the machine is loaded. Pure stdlib; silently degrades when
unsupported (non-Linux, cgroup-restricted, <16 cores, or env-disabled).
"""
import logging
import os

logger = logging.getLogger(__name__)

_ENV_DISABLE = "VLLM_OMNI_CPU_ISOLATE"
_MIN_CORES = 16


def _cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def isolate_host_thread(group: int = 0) -> bool:
    """Best-effort: raise priority + pin this thread to a core group.

    Args:
        group: logical group id (0 = orchestrator, 1/2/3 = stage cores).
    Returns:
        True if at least one isolation step succeeded.
    """
    if os.environ.get(_ENV_DISABLE, "0") == "1":
        return False
    if _cpu_count() < _MIN_CORES:
        return False
    ok = False
    try:
        os.setpriority(os.PRIO_PROCESS, 0, -10)
        ok = True
    except Exception:
        pass
    try:
        n = _cpu_count()
        base = (group % max(1, n // 4)) * 4
        os.sched_setaffinity(0, {base, base + 1, base + 2, base + 3})
        ok = True
    except Exception:
        pass
    if ok:
        logger.info("[cpu-isolate] thread group=%d affinity+priority applied", group)
    return ok
