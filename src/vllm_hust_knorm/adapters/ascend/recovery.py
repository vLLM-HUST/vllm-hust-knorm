"""Replay using real SchedulerOutput messages and the runner's forward-only RPC."""

from ...runtime.recovery import replay


def restore_request(binding, host, state, checkpoint):
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.request import RequestStatus

    class Runner:
        started = False

        def forward(self, request, tokens):
            # Admission is atomic across all layers, before allocating any page.
            managers = binding.allocator.coordinator.single_type_managers
            needed = sum(
                max(
                    0,
                    (request.layers[layer].length + len(tokens) + 127) // 128
                    - len(request.layers[layer].pages),
                )
                if request.layers
                else (len(tokens) + 127) // 128
                for layer in range(len(managers))
            )
            if needed > binding.allocator.pool.get_num_free_blocks():
                raise MemoryError("replay admission failed")
            old = request.semantic_length
            host.num_computed_tokens = old
            new_blocks = binding.engine.scheduler.kv_cache_manager.allocate_slots(
                host, len(tokens)
            )
            if new_blocks is None:
                raise MemoryError("host rejected replay allocation")
            lengths = {
                layer: request.layers[layer].length if request.layers else 0
                for layer in range(len(managers))
            }
            binding.allocator.adopt(request, lengths)
            blocks = tuple(
                [p.number for p in request.layers[layer].pages]
                for layer in range(len(managers))
            )
            known = list(checkpoint.tokens[: old + len(tokens)])
            cached = CachedRequestData.make_empty()
            if self.started:
                cached = CachedRequestData(
                    req_ids=[host.request_id],
                    resumed_req_ids=set(),
                    new_token_ids=[[]],
                    all_token_ids={},
                    new_block_ids=[new_blocks.get_block_ids()],
                    num_computed_tokens=[old],
                    num_output_tokens=[max(0, len(known) - host.num_prompt_tokens)],
                )
            output = SchedulerOutput(
                scheduled_new_reqs=[]
                if self.started
                else [NewRequestData.from_request(host, blocks)],
                scheduled_cached_reqs=cached,
                num_scheduled_tokens={host.request_id: len(tokens)},
                total_num_scheduled_tokens=len(tokens),
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[0] * len(managers),
                finished_req_ids=set(),
                free_encoder_mm_hashes=[],
            )
            binding.rpc("register", request)
            binding.rpc("replay_forward", host.request_id, known, output)
            binding.rpc("advance", request, len(tokens))
            request.tokens.extend(tokens)
            request.semantic_length += len(tokens)
            binding.allocator.adopt(
                request,
                {layer: length + len(tokens) for layer, length in lengths.items()},
            )
            binding.rpc("register", request)
            self.started = True

        def prefill(self, request, tokens):
            limit = binding.engine.vllm_config.scheduler_config.max_num_batched_tokens
            for start in range(0, len(tokens), limit):
                self.forward(request, tokens[start : start + limit])

        def decode_known(self, request, token):
            self.forward(request, [token])

    # Keep the host's delivered output and pending sampled token unchanged.
    delivered = tuple(host.output_token_ids)
    # Rebuilding storage advances its version even if no historical compression
    # occurred. Messages from before preemption must not install old pages.
    state.version += 1
    try:
        replay(state, checkpoint, binding.transactions, Runner())
        binding.rpc("restore_sampling", host.request_id, list(host.all_token_ids))
        host.num_computed_tokens = checkpoint.semantic_length
        assert tuple(host.output_token_ids) == delivered
        scheduler = binding.engine.scheduler
        scheduler.running.append(host)
        host.status = RequestStatus.RUNNING
        scheduler.reset_preempted_req_ids.discard(host.request_id)
    except Exception:
        host.num_computed_tokens = 0
        raise
