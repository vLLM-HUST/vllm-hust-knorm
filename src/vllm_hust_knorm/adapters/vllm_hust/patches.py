# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Idempotent runtime patches binding Knorm into vLLM-HUST v1 core.

The archived in-tree integration (legacy PRs #76/#134/#214) edited the
host files directly. As a plugin we install the same integration as
wrappers at the ``vllm.general_plugins`` load point:

P1  ``FreeQueue.prependleft_n`` — additive queue op (PR #76).
P2  ``BlockPool.free_blocks(prepend=)`` — head-of-queue freeing for
    uncached evicted blocks (PR #76).
P3  ``register_all_kvcache_specs`` — re-register full-attention specs
    with ``KnormFullAttentionManager`` when :func:`should_activate`
    agrees, warn when prefix caching blocks Knorm (PRs #134/#214).
    The host registry resolves this function lazily inside
    ``KVCacheSpecRegistry._ensure_registered``, and general plugins load
    before the first registration, so the module-attribute wrap is
    effective.
P4  ``Scheduler.update_from_output`` — route ``knorm_block_scores``
    from the model runner output into the manager bridge (PR #76).
P5  ``GPUModelRunner.__init__`` — compute ``_knorm_active`` from the
    same single source of truth and install the attention wrapper
    before the first forward (PR #214 guards).
P6  ``GPUModelRunner.sample_tokens`` — collect per-block scores after
    the forward and attach them to the runner output (PR #76/#214).

Every patch carries a marker and is a no-op on re-install. The patches
are inert unless the environment activates Knorm
(``VLLM_KNORM_ENABLED=1``), so a default install does not change host
behavior.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from ...knorm.config import (
    env_compression_ratio,
    env_enabled,
    should_activate,
)

_PATCHED = "_vllm_hust_knorm_patched"


def _wrap(owner: Any, name: str, builder: Callable[[Callable[..., Any]], Any]) -> bool:
    """Replace ``owner.name`` with ``builder(original)`` once.

    Returns True when this call performed the wrap, False when a
    previous install already did.
    """
    original = getattr(owner, name, None)
    if original is None:
        raise RuntimeError(
            f"vllm-hust-knorm: host surface "
            f"{getattr(owner, '__name__', owner)!s}.{name} is missing; "
            "refusing to patch (see HOST_CONTRACT.md)"
        )
    if getattr(original, _PATCHED, False):
        return False
    patched = builder(original)
    setattr(patched, _PATCHED, True)
    setattr(owner, name, patched)
    return True


def _prefix_caching_enabled(vllm_config: Any) -> bool:
    cache_config = getattr(vllm_config, "cache_config", None)
    return bool(getattr(cache_config, "enable_prefix_caching", False))


def _warn_prefix_caching_blocks_knorm() -> None:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    message = (
        "Knorm KV compression is disabled because prefix caching is "
        "enabled. Set --no-enable-prefix-caching to use Knorm."
    )
    warner = getattr(logger, "warning_once", None)
    if callable(warner):
        warner(message)
    else:  # pragma: no cover - logger drift guard
        logger.warning(message)


def _resolve_free_queue_cls(kv_cache_utils_module: Any) -> type:
    """Resolve the host free-block queue class across host generations.

    Current hosts name it ``FreeKVCacheBlockQueue``; 0.23-seam-era
    hosts named it ``FreeQueue``. Missing both is a contract breach.
    """
    for name in ("FreeKVCacheBlockQueue", "FreeQueue"):
        cls = getattr(kv_cache_utils_module, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "vllm-hust-knorm: host provides neither FreeKVCacheBlockQueue "
        "nor FreeQueue in vllm.v1.core.kv_cache_utils; refusing to "
        "patch (see HOST_CONTRACT.md)"
    )


def _queue_prepend_method(queue: Any) -> Callable[[list], None]:
    """The queue's head-insertion method, whatever the host calls it.

    Current hosts ship a native ``prepend_n``; the plugin-added
    ``prependleft_n`` covers 0.23-era queues that lack both.
    """
    for name in ("prepend_n", "prependleft_n"):
        method = getattr(queue, name, None)
        if method is not None:
            return method
    raise RuntimeError(
        "vllm-hust-knorm: free-block queue has no head-insertion "
        "method (prepend_n/prependleft_n); refusing to free with "
        "prepend"
    )


def _install_free_queue_prepend(free_queue_cls: type) -> bool:
    """P1: ensure the queue supports head insertion (idempotent).

    Hosts with a native ``prepend_n`` (current main) or
    ``prependleft_n`` (0.23 seam era) need nothing; older shapes get
    the in-tree ``prependleft_n`` implementation added (legacy PR #76).
    """
    has_native = any(
        callable(getattr(free_queue_cls, name, None))
        for name in ("prepend_n", "prependleft_n")
    )
    if has_native:
        return False

    def prependleft_n(self: Any, blocks: list[Any]) -> None:
        """Insert blocks at the head of the free queue.

        These blocks will be allocated first by subsequent
        :meth:`popleft_n` calls.
        """
        if not blocks:
            return

        old_first = self.fake_free_list_head.next_free_block

        # Add inter-connections between consecutive blocks.
        for i in range(len(blocks) - 1):
            blocks[i].next_free_block = blocks[i + 1]
            blocks[i + 1].prev_free_block = blocks[i]

        # Wire fake_head → blocks[0]; blocks[-1] closes the list.
        self.fake_free_list_head.next_free_block = blocks[0]
        blocks[0].prev_free_block = self.fake_free_list_head
        if old_first is None:
            # Degenerate empty queue (never happens on a real host,
            # whose queue starts fully populated) — close the list.
            blocks[-1].next_free_block = None
        else:
            blocks[-1].next_free_block = old_first
            old_first.prev_free_block = blocks[-1]

        self.num_free_blocks += len(blocks)

    free_queue_cls.prependleft_n = prependleft_n
    return True


def _install_free_blocks_prepend(block_pool_cls: type) -> bool:
    """P2: teach ``free_blocks`` the ``prepend`` keyword.

    Hosts that already ship the in-tree signature (0.23 seam era) are
    detected by signature and left untouched.
    """
    original = block_pool_cls.free_blocks
    if getattr(original, _PATCHED, False):
        return False
    # functools.wraps makes inspect.signature unwrap to the original,
    # so the marker check above must come first; this signature probe
    # only detects hosts that natively ship the prepend keyword.
    if "prepend" in inspect.signature(original).parameters:
        return False

    @functools.wraps(original)
    def free_blocks(self: Any, ordered_blocks: Any, prepend: bool = False) -> None:
        """Free blocks; ``prepend=True`` re-uses them first."""
        if not prepend:
            original(self, ordered_blocks)
            return
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        freed = [
            block for block in blocks_list if block.ref_cnt == 0 and not block.is_null
        ]
        if freed:
            _queue_prepend_method(self.free_block_queue)(freed)

    setattr(free_blocks, _PATCHED, True)
    block_pool_cls.free_blocks = free_blocks
    return True


def _override_full_attention_specs(registry_module: Any) -> list[str]:
    """Re-register host full-attention specs with the Knorm manager.

    Every spec the host mapped to ``FullAttentionManager`` under the
    ``FullAttentionSpec`` uniform group moves to
    ``KnormFullAttentionManager``. Re-registration goes through the
    module-private registry dict (pop-then-register) because the public
    ``register`` rejects conflicting updates — the same idiom the
    legacy host tests used. Returns the redirected spec class names
    (used by the log marker below).
    """
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from ...knorm.manager import get_knorm_manager_class

    registry = registry_module._REGISTRY_KVCACHESPEC_LIST
    knorm_manager = get_knorm_manager_class()
    targets = [
        spec
        for spec, meta in registry.items()
        if meta.manager_class is FullAttentionManager
        and meta.uniform_type_base_spec is FullAttentionSpec
    ]
    for spec in targets:
        registry.pop(spec)
        registry_module.KVCacheSpecRegistry.register(
            spec,
            knorm_manager,
            uniform_type_base_spec=FullAttentionSpec,
        )
    return [spec.__name__ for spec in targets]


def _install_spec_registration_patch(
    single_type_module: Any, registry_module: Any
) -> bool:
    """P3: wrap ``register_all_kvcache_specs``.

    Manager registration follows :func:`should_activate` — the same
    single source of truth the runner-side guard uses — so the
    half-enabled state of legacy issue #163 cannot recur.
    """

    def build(original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def register_all_kvcache_specs(vllm_config: Any) -> None:
            original(vllm_config)
            if vllm_config is None:
                return
            prefix_caching = _prefix_caching_enabled(vllm_config)
            if not should_activate(prefix_caching):
                if prefix_caching and env_enabled() and env_compression_ratio() < 1:
                    _warn_prefix_caching_blocks_knorm()
                return
            redirected = _override_full_attention_specs(registry_module)
            print(
                "[vllm-hust-knorm] KnormFullAttentionManager registered "
                f"for {', '.join(redirected)} "
                "(log marker: knorm-manager-registered)",
                flush=True,
            )

        return register_all_kvcache_specs

    return _wrap(single_type_module, "register_all_kvcache_specs", build)


def _install_scheduler_score_routing(scheduler_cls: type) -> bool:
    """P4: route ``knorm_block_scores`` into the manager bridge."""
    from ...knorm.manager import submit_block_scores

    def build(original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def update_from_output(
            self: Any, scheduler_output: Any, model_runner_output: Any
        ) -> Any:
            knorm_scores = getattr(model_runner_output, "knorm_block_scores", None)
            if knorm_scores:
                submit_block_scores(knorm_scores)
            return original(self, scheduler_output, model_runner_output)

        return update_from_output

    return _wrap(scheduler_cls, "update_from_output", build)


def _install_runner_init_patch(runner_cls: type) -> bool:
    """P5: compute ``_knorm_active`` once and install the wrapper early."""

    def build(original: Callable[..., Any]) -> Callable[..., Any]:
        from ...knorm.attention_backend import install_attention_wrapper

        @functools.wraps(original)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)
            self._knorm_scores = None
            self._knorm_wrapper_installed = False
            # self.cache_config is a CacheConfig (not a VllmConfig), so
            # read the flag directly instead of _prefix_caching_enabled.
            prefix_caching = bool(
                getattr(
                    getattr(self, "cache_config", None),
                    "enable_prefix_caching",
                    False,
                )
            )
            self._knorm_active = should_activate(prefix_caching)
            if self._knorm_active:
                impl = install_attention_wrapper()
                self._knorm_wrapper_installed = True
                print(
                    "[vllm-hust-knorm] attention wrapper installed on "
                    f"{impl} (log marker: knorm-wrapper-installed)",
                    flush=True,
                )

        return __init__

    return _wrap(runner_cls, "__init__", build)


def _is_last_pp_rank() -> bool:
    """True on the last pipeline-parallel rank (or without PP groups)."""
    try:
        from vllm.distributed import get_pp_group

        return get_pp_group().is_last_rank
    except Exception:  # pragma: no cover - single-process default
        return True


def _install_runner_sample_patch(runner_cls: type) -> bool:
    """P6: collect scores after the forward, attach to the output."""

    def build(original: Callable[..., Any]) -> Callable[..., Any]:
        from ...knorm.hooks import attach_knorm_scores, collect_knorm_scores

        @functools.wraps(original)
        def sample_tokens(self: Any, grammar_output: Any = None) -> Any:
            if getattr(self, "_knorm_wrapper_installed", False) and (
                _is_last_pp_rank()
            ):
                collect_knorm_scores(self, self.input_batch)
            result = original(self, grammar_output)
            if getattr(self, "_knorm_wrapper_installed", False):
                attach_knorm_scores(self, result)
            return result

        return sample_tokens

    return _wrap(runner_cls, "sample_tokens", build)


def install_patches() -> dict[str, Any]:
    """Install all host patches (idempotent) and report what happened.

    Raises RuntimeError when a required host surface is missing — never
    silently degrade.
    """
    import vllm.v1.core.block_pool as block_pool_module
    import vllm.v1.core.kv_cache_utils as kv_cache_utils_module
    import vllm.v1.core.sched.scheduler as scheduler_module
    import vllm.v1.core.single_type_kv_cache_manager as stm_module
    import vllm.v1.kv_cache_spec_registry as registry_module
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    free_queue_cls = _resolve_free_queue_cls(kv_cache_utils_module)
    applied = {
        "free_queue_head_insertion": _install_free_queue_prepend(free_queue_cls),
        "block_pool_free_blocks_prepend": _install_free_blocks_prepend(
            block_pool_module.BlockPool
        ),
        "spec_registration": _install_spec_registration_patch(
            stm_module, registry_module
        ),
        "scheduler_score_routing": _install_scheduler_score_routing(
            scheduler_module.Scheduler
        ),
        "runner_init": _install_runner_init_patch(GPUModelRunner),
        "runner_sample_tokens": _install_runner_sample_patch(GPUModelRunner),
    }
    return {
        "patches": applied,
        "already_installed": not any(applied.values()),
        "activation": "set VLLM_KNORM_ENABLED=1 to activate (see config)",
    }


__all__ = ["install_patches"]
