# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Score centering for training-inference mismatch (https://arxiv.org/abs/2609.20807).

The sampler's top-k head is exact; its tail is modeled as the trainer's tail rescaled to the
sampler's tail mass, so the centering term only needs the k head log-probs of the trainer.
"""

import math
from collections.abc import Callable
from typing import Optional

import torch
import torch.distributed as dist

from verl.trainer.ppo.rollout_corr_helper import _parse_rollout_is_threshold

ROLLOUT_TOPK_FIELDS = ("rollout_topk_ids", "rollout_topk_log_probs")
TOPK_LOG_PROB_CHUNK_SIZE = 4096


def dummy_rollout_topk(num_rows: int, k: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a uniform top-k head for rows the loss masks out (e.g. prompt or padding positions).

    Args:
        num_rows: Number of rows to fill.
        k: Head size.
        device: Device for the returned tensors, default None (CPU).

    Returns:
        Tuple containing:
            ids: Token ids 0..k-1 repeated per row, shape (num_rows, k), dtype int32.
            log_probs: Uniform log-probs -log(k), shape (num_rows, k), dtype float32.
    """
    ids = torch.arange(k, dtype=torch.int32, device=device).expand(num_rows, k).clone()
    log_probs = torch.full((num_rows, k), -math.log(k), dtype=torch.float32, device=device)
    return ids, log_probs


def pad_rollout_topk(
    response_topk_ids,
    response_topk_log_probs,
    *,
    k: int,
    prompt_width: int,
    response_width: int,
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay the sampler's response-token heads over the padded full sequence.

    Prompts are left-padded to ``prompt_width`` and responses are right-padded to
    ``response_width``, so response token r sits at sequence column ``prompt_width + r``
    and is predicted by the logits row ``prompt_width - 1 + r``; its sampler head goes there.
    Rows outside the response (prompt and padding) get the uniform dummy head.

    Args:
        response_topk_ids: Sampler top-k token ids for the response tokens, shape
            (response_length, k), array-like of ints.
        response_topk_log_probs: Sampler top-k log-probs matching ``response_topk_ids``,
            shape (response_length, k), array-like of floats.
        k: Head size.
        prompt_width: Padded prompt length.
        response_width: Padded response length.
        response_length: Number of real (non-padding) response tokens.

    Returns:
        Tuple containing:
            ids: Full-sequence top-k ids, shape (1, prompt_width + response_width, k), dtype int32.
            log_probs: Full-sequence top-k log-probs matching ``ids``, same shape, dtype float32.
    """
    ids = torch.as_tensor(response_topk_ids, dtype=torch.int32)[:response_length]
    log_probs = torch.as_tensor(response_topk_log_probs, dtype=torch.float32)[:response_length]
    if ids.shape != (response_length, k) or log_probs.shape != (response_length, k):
        raise ValueError(
            f"score centering needs one sampler top-{k} head per response token, "
            f"got {tuple(ids.shape)} for {response_length} tokens"
        )
    full_ids, full_log_probs = dummy_rollout_topk(prompt_width + response_width, k)
    start = prompt_width - 1
    full_ids[start : start + response_length] = ids
    full_log_probs[start : start + response_length] = log_probs
    return full_ids.unsqueeze(0), full_log_probs.unsqueeze(0)


def _maybe_all_reduce(tensor: torch.Tensor, op, process_group) -> None:
    if process_group is not None and dist.is_initialized():
        dist.all_reduce(tensor, op=op, group=process_group)


class _TopKLogProbsFromLogits(torch.autograd.Function):
    """log_softmax(logits).gather(ids) with a bounded fp32 workspace in forward and backward.

    Only the original logits, the ids and two scalars per row are saved; the softmax is
    recomputed per chunk in backward. The vocab axis may be sharded across process_group.
    """

    @staticmethod
    def forward(ctx, logits, token_ids, process_group, chunk_size):
        n, vocab_size = logits.shape
        rank = dist.get_rank(process_group) if process_group is not None and dist.is_initialized() else 0
        vocab_start = rank * vocab_size
        chunk_size = chunk_size if chunk_size > 0 else max(n, 1)
        output = torch.empty(token_ids.shape, device=logits.device, dtype=torch.float32)
        maxima = torch.empty((n, 1), device=logits.device, dtype=torch.float32)
        denominators = torch.empty_like(maxima)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            work = logits[start:end].to(dtype=torch.float32, copy=True)
            maximum = work.max(dim=-1, keepdim=True).values
            _maybe_all_reduce(maximum, dist.ReduceOp.MAX, process_group)
            ids = token_ids[start:end]
            local_mask = (ids >= vocab_start) & (ids < vocab_start + vocab_size)
            local_ids = (ids - vocab_start).clamp(0, vocab_size - 1)
            targets = work.gather(-1, local_ids).masked_fill_(~local_mask, 0.0)
            _maybe_all_reduce(targets, dist.ReduceOp.SUM, process_group)
            work.sub_(maximum).exp_()
            denominator = work.sum(dim=-1, keepdim=True)
            _maybe_all_reduce(denominator, dist.ReduceOp.SUM, process_group)
            output[start:end] = (targets - maximum) - denominator.log()
            maxima[start:end] = maximum
            denominators[start:end] = denominator
            del work
        ctx.save_for_backward(logits, token_ids, maxima, denominators)
        ctx.chunk_size = chunk_size
        ctx.vocab_start = vocab_start
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, token_ids, maxima, denominators = ctx.saved_tensors
        n, vocab_size = logits.shape
        grad_input = torch.empty_like(logits)
        for start in range(0, n, ctx.chunk_size):
            end = min(start + ctx.chunk_size, n)
            work = logits[start:end].to(dtype=torch.float32, copy=True)
            work.sub_(maxima[start:end]).exp_().div_(denominators[start:end])
            grad = grad_output[start:end].float()
            work.mul_(-grad.sum(dim=-1, keepdim=True))
            ids = token_ids[start:end]
            local_mask = (ids >= ctx.vocab_start) & (ids < ctx.vocab_start + vocab_size)
            local_ids = (ids - ctx.vocab_start).clamp(0, vocab_size - 1)
            work.scatter_add_(-1, local_ids, grad.masked_fill(~local_mask, 0.0))
            grad_input[start:end] = work
            del work
        return grad_input, None, None, None


def topk_log_probs_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    chunk_size: int = TOPK_LOG_PROB_CHUNK_SIZE,
    process_group=None,
) -> torch.Tensor:
    """Compute full-vocab-normalized log-probs of ``logits`` at ``token_ids``, chunked for memory.

    Equivalent to ``torch.log_softmax(logits, dim=-1).gather(-1, token_ids)`` but never
    materializes the full (N, V) softmax at once: the normalizer is recomputed per chunk in
    forward and backward, bounding the extra fp32 workspace to (chunk_size, V).

    Args:
        logits: Trainer logits, shape (N, V), any float dtype.
        token_ids: Token ids to gather, shape (N, k), any integer dtype.
        chunk_size: Number of rows processed per chunk. Default: 4096.
        process_group: Optional process group if the vocab dimension is sharded across ranks,
            in which case each rank holds a local shard of size V and ``token_ids`` are global
            ids. Default: None (no sharding).

    Returns:
        Log-probs of ``token_ids`` under the full-vocab softmax of ``logits``, shape (N, k),
        dtype float32, with gradient to ``logits``.
    """
    return _TopKLogProbsFromLogits.apply(logits, token_ids.long(), process_group, chunk_size)


def score_centering_weight_fn(
    rollout_is: Optional[str], rollout_is_threshold
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the token-level IS rule as a function of a ratio, shared by the head and sampled token.

    Args:
        rollout_is: Importance-sampling mode score centering composes with: None (no IS weight)
            or "token" (TIS or IcePop, depending on ``rollout_is_threshold``).
        rollout_is_threshold: Threshold specification, see ``_parse_rollout_is_threshold``: a
            single float or float-like string upper-clamps (TIS); a "lower_upper" string zeros
            ratios outside the band (IcePop).

    Returns:
        A function mapping a ratio tensor to its weight tensor of the same shape.
    """
    if rollout_is is None:
        return torch.ones_like
    if rollout_is != "token":
        raise ValueError(f"score centering composes with token-level IS only, got rollout_is={rollout_is!r}")
    upper, lower = _parse_rollout_is_threshold(rollout_is_threshold)
    if lower is None:
        return lambda ratio: ratio.clamp(max=upper)
    return lambda ratio: torch.where((ratio >= lower) & (ratio <= upper), ratio, torch.zeros_like(ratio))


def score_centering_correction(
    train_head_log_probs: torch.Tensor,
    sampler_head_log_probs: torch.Tensor,
    weight_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-token centering term ``sum_H sg[q w - alpha p] log p`` and the head masses.

    The sampler's tail (outside the top-k head H) is modeled as the trainer's tail rescaled by
    ``alpha = rho * weight_fn(1 / rho)`` with ``rho = (1 - q_mass) / (1 - p_mass)``, so the
    correction only needs the k head log-probs of both distributions. When the trainer has no
    tail (``p_mass`` saturates to 1, e.g. the head spans the whole vocabulary), ``alpha`` is
    forced to 0 instead of the ill-conditioned ``rho``.

    Args:
        train_head_log_probs: Trainer's full-vocab-normalized log-probs at the head ids,
            shape (N, k), any float dtype, with gradient to the trainer logits.
        sampler_head_log_probs: Sampler's full-vocab-normalized log-probs at the same head ids,
            shape (N, k), any float dtype.
        weight_fn: IS weight rule applied to the ratio p/q on the head, as returned by
            ``score_centering_weight_fn``.
        eps: Numerical floor for the tail masses in ``rho``. Default: 1e-6.

    Returns:
        Tuple containing:
            correction: Centering term to subtract from the policy-gradient loss, shape (N,),
                dtype float32, with gradient to the trainer logits.
            sampler_head_mass: Sampler's head probability mass, shape (N,), dtype float32.
            train_head_mass: Trainer's head probability mass, shape (N,), dtype float32.
    """
    train_head_log_probs = train_head_log_probs.float()
    with torch.no_grad():
        p = train_head_log_probs.exp()
        q = sampler_head_log_probs.float().exp()
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        train_tail_mass = (1 - p_mass).clamp_min(0.0)
        sampler_tail_mass = (1 - q_mass).clamp_min(0.0)
        rho = sampler_tail_mass.clamp_min(eps) / train_tail_mass.clamp_min(eps)
        # No trainer tail to rescale (e.g. the head already spans the full vocabulary):
        # the tail model contributes nothing, whatever rho's ill-conditioned ratio evaluates to.
        alpha = torch.where(train_tail_mass > eps, rho * weight_fn(1.0 / rho), torch.zeros_like(rho))
        head_weights = weight_fn((train_head_log_probs - sampler_head_log_probs.float()).exp())
        residual = q * head_weights - alpha.unsqueeze(-1) * p
    correction = (residual * train_head_log_probs).sum(-1)
    return correction, q_mass, p_mass
