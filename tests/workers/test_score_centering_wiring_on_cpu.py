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

"""CPU coverage for score centering's worker wiring: the actor loss-fn selection and the FSDP engine outputs."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.score_centering import score_centering_ppo_loss
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import select_actor_loss_fn
from verl.workers.utils.losses import ppo_loss


def _actor_config(**kwargs):
    return ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, **kwargs)


def test_select_actor_loss_fn_picks_score_centering():
    config = _actor_config(
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=RolloutCorrectionConfig.bypass_pg_sc())
    )
    fn = select_actor_loss_fn(config, distillation_config=None)
    assert isinstance(fn, partial) and fn.func is score_centering_ppo_loss

    fn = select_actor_loss_fn(_actor_config(), distillation_config=None)
    assert fn.func is ppo_loss


def test_select_actor_loss_fn_rejects_fused_kernels_for_score_centering():
    config = _actor_config(
        use_fused_kernels=True,
        policy_loss=PolicyLossConfig(
            loss_mode="bypass_mode", rollout_correction=RolloutCorrectionConfig.bypass_pg_sc()
        ),
    )
    with pytest.raises(NotImplementedError):
        select_actor_loss_fn(config, distillation_config=None)


@pytest.mark.parametrize("strategy", ["megatron", "veomni"])
def test_select_actor_loss_fn_rejects_non_fsdp_for_score_centering(strategy):
    config = ActorConfig(
        strategy=strategy,
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(
            loss_mode="bypass_mode", rollout_correction=RolloutCorrectionConfig.bypass_pg_sc()
        ),
    )
    with pytest.raises(NotImplementedError, match="FSDP engine only"):
        select_actor_loss_fn(config, distillation_config=None)


def test_select_actor_loss_fn_rejects_distillation_with_score_centering():
    config = _actor_config(
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=RolloutCorrectionConfig.bypass_pg_sc())
    )
    with pytest.raises(ValueError, match="distillation"):
        select_actor_loss_fn(config, distillation_config=SimpleNamespace(enabled=True))


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("score_centering", [True, False])
def test_prepare_model_outputs_keeps_logits_intact_for_score_centering(use_remove_padding, score_centering):
    """The score centering hook saves the logits and re-reads them in its backward, so the engine
    must not run the flash-attn cross-entropy backward that writes the gradient into the logits."""
    seq_lengths = torch.tensor([3, 2])
    total_nnz, vocab_size = int(seq_lengths.sum()), 8
    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)])
    input_ids = torch.nested.nested_tensor_from_jagged(torch.randint(0, vocab_size, (total_nnz,)), offsets=cu_seqlens)
    output = type("Output", (), {})()
    output_args = {"input_ids_rmpad_rolled": torch.randint(0, vocab_size, (total_nnz,))}
    if use_remove_padding:
        output.logits = torch.randn(1, total_nnz, vocab_size)
        output_args.update(temperature_rmpad=torch.ones(total_nnz), pad_size=0)
    else:
        output.logits = torch.randn(2, 3, vocab_size)
        output_args["temperature"] = torch.ones(2)

    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        score_centering=score_centering,
    )
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False

    def logits_processor(student_logits, data):
        return {"sc_correction": torch.zeros(student_logits.shape[:2])}

    with patch(
        "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits", return_value=torch.zeros(total_nnz)
    ) as mock_logprobs:
        FSDPEngineWithLMHead.prepare_model_outputs(engine, output, output_args, micro_batch, logits_processor)

    assert mock_logprobs.call_args.kwargs.get("inplace_backward", True) is not score_centering
