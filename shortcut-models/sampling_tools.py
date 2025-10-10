import torch
import numpy as np
from typing import Iterable, List, Tuple
from torch_model import DiT
from typing import List, Tuple, Iterable, Optional
from collections import deque
from fractions import Fraction

def denoising_step(model, x, ti, denoise_timesteps, labels, cfg_scale, num_classes, device):
    batch_size = x.shape[0]
    delta_t = 1.0 / denoise_timesteps
    t = ti / denoise_timesteps
    t_vector = torch.full((batch_size,), t, dtype=torch.float32, device=device)
    dt_flow = int(np.log2(denoise_timesteps))
    dt_base = torch.ones(batch_size, dtype=torch.float32, device=device) * dt_flow

    # Classifier-free guidance
    labels_uncond = torch.ones_like(labels) * num_classes
    if cfg_scale == 1:
        v, *_ = model(x, t_vector, dt_base, labels)
    elif cfg_scale == 0:
        if isinstance(model, DiT):
            v, *_ = model(x, t_vector, dt_base, labels_uncond)
        else:
            v, *_ = model.apply_model(x, t_vector, dt_base, labels_uncond)
             
    else:
        if isinstance(model, DiT):
            v_pred_uncond, *_ = model(x, t_vector, dt_base, labels_uncond)
            v_pred_label, *_ = model(x, t_vector, dt_base, labels)
        else:
            v_pred_uncond, *_ = model.apply_model(x, t_vector, dt_base, labels_uncond)
            v_pred_label, *_ = model.apply_model(x, t_vector, dt_base, labels)
        v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

    # Euler sampling
    x = x + v * delta_t
    return x


def build_step_schedule(fractions: Iterable[float],
                        step_sizes: Iterable[int],
                        *,
                        tol: float = 1e-8) -> Tuple[List[int], List[int]]:
    """
    Build two parallel lists (indices, totals) so that calling step(idx, tot)
    for each zipped pair traverses the path described by `fractions` using the
    provided `step_sizes` for each segment.

    Args:
        fractions: sequence of positive floats that sum to 1.0
        step_sizes: sequence of positive integers (e.g., 1,2,4,8,16,32,64,128).
                    Same length as fractions. Each fraction * step_size must be an integer.
        tol: tolerance for floating comparisons.

    Returns:
        (indices, totals) where len(indices) == len(totals). For each k,
        call step(indices[k], totals[k]).

    Example:
        fractions = (0.5, 0.25, 0.25)
        step_sizes = (2, 64, 128)

        -> indices = [0, 32, 33, ..., 47, 96, ..., 127]
           totals  = [2, 64, 64, ..., 64, 128, ..., 128]
    """
    if isinstance(fractions, int):
        fracs = [fractions]
    else:
        fracs = list(fractions)
    if isinstance(step_sizes, int):
        sizes = [step_sizes]
    else:
        sizes = list(step_sizes)
        
    if len(fracs) == 0:
        raise ValueError("fractions must be non-empty.")
    if len(fracs) != len(sizes):
        raise ValueError("fractions and step_sizes must have the same length.")

    # basic validity
    if any(f <= 0 for f in fracs):
        raise ValueError("All fractions must be positive.")
    if any(s <= 0 or not isinstance(s, int) for s in sizes):
        raise ValueError("All step_sizes must be positive integers.")

    ssum = sum(fracs)
    if not (abs(ssum - 1.0) <= tol):
        raise ValueError(f"fractions must sum to 1.0 (sum={ssum})")

    indices: List[int] = []
    totals: List[int] = []

    cumulative_frac = 0.0
    for f, S in zip(fracs, sizes):
        # number of steps in this segment for the given granularity S
        exact_n = f * S
        n = int(round(exact_n))
        if not (abs(exact_n - n) <= tol):
            raise ValueError(f"fraction {f} * step_size {S} must be integer (got {exact_n})")
        if n == 0:
            # allow zero-length segment only if f == 0 (we checked f>0 above), so error
            raise ValueError(f"Segment with fraction {f} and step_size {S} results in zero steps.")
        # start index in [0, S-1] (inclusive)
        start = int(round(cumulative_frac * S))
        end = start + n  # exclusive
        if end > S:
            # numerical safety: trim to S
            end = S
        # add indices start .. end-1 with total S
        for idx in range(start, end):
            indices.append(idx)
            totals.append(S)

        # advance cumulative fraction
        cumulative_frac += f

    # final checks: for the last segment, ensure last index equals S-1 of its S
    # (this should hold by construction if fractions sum to 1 and integer counts matched)
    if len(indices) == 0:
        return indices, totals

    # Remove possible duplicates and ensure monotonic within each (index,total) pair sequence:
    # (we preserve order; duplicates unlikely but handle gracefully)
    cleaned_idx = []
    cleaned_tot = []
    seen = set()
    for idx, tot in zip(indices, totals):
        key = (idx, tot)
        if key in seen:
            continue
        seen.add(key)
        cleaned_idx.append(idx)
        cleaned_tot.append(tot)

    return cleaned_idx, cleaned_tot



def shortest_plan_to_end(current_index: int,
                                 current_total: int,
                                 allowed_totals: Optional[Iterable[int]] = None
                                ) -> Tuple[List[int], List[int]]:
    """
    Compute minimal sequence of aligned steps to reach p == 1 starting from
    p = current_index / current_total under the rule:
      - step(k, T) is allowed only when p == k / T (i.e. p * T is integer)
      - calling step(k, T) moves p -> (k + 1) / T

    Args:
      current_index: int, 0 <= current_index < current_total
      current_total: int, dyadic total (power of two) that current_index refers to
      allowed_totals: iterable of allowed totals (default: [2,4,8,16,32,64,128])

    Returns:
      (indices, totals): two parallel lists describing the calls in order.
    """
    if allowed_totals is None:
        allowed_totals = [2, 4, 8, 16, 32, 64, 128]
    allowed = sorted(set(allowed_totals))

    if current_total not in allowed:
        raise ValueError("current_total must be one of allowed_totals")

    if current_total < 2 or (current_total & (current_total - 1)) != 0:
        raise ValueError("current_total must be a power of two >= 2")
    if not (0 <= current_index < current_total):
        raise ValueError("current_index must be in [0, current_total-1]")

    start_p = Fraction(current_index, current_total)
    target = Fraction(1, 1)

    # Quick exit if somehow already at or beyond end
    if start_p >= target:
        return [], []

    # BFS over reachable p values (rationals of form (k+1)/T)
    q = deque([start_p])
    prev = {start_p: (None, None)}  # map p -> (prev_p, (idx, T))
    visited = {start_p}

    while q:
        p = q.popleft()
        # for each allowed grid T, only allow a step if p * T is integer
        for T in allowed:
            # check alignment: p * T must be integer
            if (p.numerator * T) % p.denominator != 0:
                continue
            k = (p.numerator * T) // p.denominator  # exact integer
            # k must be in [0, T-1] since p < 1
            if not (0 <= k < T):
                continue
            p_next = Fraction(k + 1, T)  # result of step(k,T)
            # only forward progress (should always be > p though)
            if p_next <= p:
                continue
            if p_next in visited:
                continue
            prev[p_next] = (p, (k, T))
            if p_next == target:
                # reconstruct path
                inds: List[int] = []
                tots: List[int] = []
                cur = p_next
                while True:
                    pprev, action = prev[cur]
                    if pprev is None:
                        break
                    k_act, T_act = action
                    inds.append(k_act)
                    tots.append(T_act)
                    cur = pprev
                inds.reverse()
                tots.reverse()
                return inds, tots
            visited.add(p_next)
            q.append(p_next)

    # If we cannot reach p == 1 with these allowed totals, raise.
    raise RuntimeError("Unable to reach end with the given allowed_totals")
