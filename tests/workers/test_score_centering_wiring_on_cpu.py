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

"""CPU coverage for the actor loss-fn selection: distillation, score centering, and plain PPO."""

from functools import partial

import pytest

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.score_centering import score_centering_ppo_loss
from verl.workers.config import ActorConfig, PolicyLossConfig
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


def test_select_actor_loss_fn_rejects_megatron_for_score_centering():
    config = ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(
            loss_mode="bypass_mode", rollout_correction=RolloutCorrectionConfig.bypass_pg_sc()
        ),
    )
    with pytest.raises(NotImplementedError):
        select_actor_loss_fn(config, distillation_config=None)
