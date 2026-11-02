"""Device-only K scoring; only selected indices/position sidecars reach control CPU."""


def select_tensor(keys, positions, keep):
    import torch

    # [head, token, dim]. Explicit reduction order matches the CPU FP32 oracle.
    if keys.ndim != 3 or positions.shape != keys.shape[:2]:
        raise ValueError("invalid selection tensor shape")
    if not 0 < keep <= keys.shape[1]:
        raise ValueError("invalid keep count")
    scores = torch.zeros(keys.shape[:2], dtype=torch.float32, device=keys.device)
    for dim in range(keys.shape[2]):
        component = keys[:, :, dim].float()
        scores = scores + component * component
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("non-finite K score")
    by_position = torch.argsort(positions, dim=1, stable=True)
    ordered_positions = positions.gather(1, by_position)
    if bool((ordered_positions[:, 1:] == ordered_positions[:, :-1]).any()):
        raise ValueError("duplicate original positions")
    by_score = torch.argsort(scores.gather(1, by_position), dim=1, stable=True)
    return by_position.gather(1, by_score[:, :keep])
