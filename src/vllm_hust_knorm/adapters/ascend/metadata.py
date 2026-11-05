"""Rewrite physical attention coordinates while preserving semantic positions."""

from copy import copy


def replace_block_tables(runner, request, layouts):
    state = runner.requests[request.request_id]
    tables = tuple(
        [p.number for p in layouts[layer].pages] for layer in sorted(layouts)
    )
    state.block_ids = tables
    batch = runner.input_batch
    row = batch.req_id_to_index.get(request.request_id)
    if row is not None:
        # Removal can leave holes until the runner condenses the batch. Preserve
        # its row/sampling associations and copy through the last live row.
        rows = max(batch.req_id_to_index.values()) + 1
        for group, blocks in enumerate(tables):
            table = batch.block_table[group]
            # add_row alone does not erase the formerly longer tail.
            table.clear_row(row)
            table.add_row(blocks, row)
            table.commit_block_table(rows)


def physical_metadata(runner, group, common, states):
    import torch

    result = copy(common)
    lengths, computed, slots = [], [], []
    starts = common.query_start_loc_cpu.tolist()
    for row, req_id in enumerate(runner.input_batch.req_ids):
        query = starts[row + 1] - starts[row]
        state = states.get(req_id)
        semantic = int(runner.input_batch.num_computed_tokens_cpu[row])
        length = semantic if state is None else state.layers[group].length
        if state is not None and state.phase == "decode" and query != 1:
            raise RuntimeError("compressed histories support single-token decode only")
        table = runner.input_batch.block_table[group].get_cpu_tensor()[row]
        computed.append(length)
        lengths.append(length + query)
        slots.extend(
            int(table[i // 128]) * 128 + i % 128 for i in range(length, length + query)
        )
    padding = common.num_reqs - len(lengths)
    lengths += [0] * padding
    computed += [0] * padding
    cpu_lengths = torch.tensor(lengths, dtype=torch.int32)
    result.seq_lens = cpu_lengths.to(common.seq_lens.device)
    result._seq_lens_cpu = cpu_lengths
    result.seq_lens_cpu = cpu_lengths
    result.seq_lens_cpu_upper_bound = cpu_lengths
    result.num_computed_tokens_cpu = torch.tensor(computed, dtype=torch.int32)
    result.max_seq_len = max(lengths, default=0)
    result.slot_mapping = torch.tensor(
        slots + [-1] * (common.num_input_tokens - len(slots)),
        dtype=torch.int64,
        device=common.slot_mapping.device,
    )
    # Semantic positions and is_prefilling are intentionally inherited.
    return result
