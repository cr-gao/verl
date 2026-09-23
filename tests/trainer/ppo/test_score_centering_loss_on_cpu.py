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

"""CPU coverage for score centering's loss integration: REINFORCE, bypass dispatch, and the hook wrapper."""

import math

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.core_algos import compute_policy_loss_bypass_mode, compute_policy_loss_reinforce
from verl.trainer.ppo.score_centering import score_centering_logits_processor, score_centering_ppo_loss
from verl.workers.config import ActorConfig, PolicyLossConfig


def _actor_config(rollout_correction):
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=rollout_correction),
        loss_agg_mode="token-mean",
    )
    config.global_batch_info.update(dp_size=1, batch_num_tokens=None, global_batch_size=None, loss_scale_factor=None)
    return config


def test_reinforce_adds_advantage_times_correction():
    torch.manual_seed(0)
    log_prob = torch.randn(2, 4, requires_grad=True)
    rollout_log_prob = log_prob.detach() + 0.1
    advantages = torch.randn(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    correction = torch.randn(2, 4, requires_grad=True)
    config = _actor_config(RolloutCorrectionConfig.bypass_pg_sc())
    loss_sc, metrics = compute_policy_loss_reinforce(
        rollout_log_prob, log_prob, advantages, mask, "token-mean", config, sc_correction=correction
    )
    loss_plain, _ = compute_policy_loss_reinforce(rollout_log_prob, log_prob, advantages, mask, "token-mean", config)
    torch.testing.assert_close(loss_sc, loss_plain + (advantages * correction).mean())
    assert math.isclose(metrics["actor/sc_correction"], correction.mean().item(), rel_tol=1e-5)


def test_bypass_mode_dispatches_correction_to_reinforce_only():
    torch.manual_seed(1)
    log_prob = torch.randn(2, 3, requires_grad=True)
    old = log_prob.detach()
    adv = torch.randn(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    corr = torch.randn(2, 3)
    config = _actor_config(RolloutCorrectionConfig.bypass_pg_sc())
    loss, metrics = compute_policy_loss_bypass_mode(old, log_prob, adv, mask, "token-mean", config, sc_correction=corr)
    assert "actor/sc_correction" in metrics
    config = _actor_config(RolloutCorrectionConfig.bypass_ppo_clip())
    with pytest.raises(ValueError, match="score centering"):
        compute_policy_loss_bypass_mode(old, log_prob, adv, mask, "token-mean", config, sc_correction=corr)


def _nested_batch(lengths, k, vocab):
    ids = torch.nested.as_nested_tensor(
        [torch.randint(0, vocab, (n, k), dtype=torch.int32) for n in lengths], layout=torch.jagged
    )
    log_probs = torch.nested.as_nested_tensor(
        [
            torch.log_softmax(torch.randn(n, vocab), dim=-1).gather(-1, i.long())
            for n, i in zip(lengths, ids.unbind(), strict=False)
        ],
        layout=torch.jagged,
    )
    data = TensorDict({"rollout_topk_ids": ids, "rollout_topk_log_probs": log_probs}, batch_size=[len(lengths)])
    return data


def test_logits_processor_returns_per_token_scalars():
    torch.manual_seed(2)
    lengths, k, vocab = [3, 5], 4, 17
    data = _nested_batch(lengths, k, vocab)
    logits = torch.randn(1, sum(lengths), vocab, requires_grad=True)
    config = _actor_config(RolloutCorrectionConfig.bypass_pg_sc())
    out = score_centering_logits_processor(student_logits=logits, data=data, config=config)
    assert set(out) == {"sc_correction", "sc_sampler_head_mass", "sc_train_head_mass"}
    for v in out.values():
        assert v.shape == (1, sum(lengths))
    assert out["sc_correction"].requires_grad
    assert (out["sc_sampler_head_mass"] <= 1.0 + 1e-6).all()


def test_score_centering_ppo_loss_routes_hook_and_final_loss(monkeypatch):
    config = _actor_config(RolloutCorrectionConfig.bypass_pg_sc())
    called = {}

    def fake_ppo_loss(config, model_output, data, dp_group=None):
        called["ppo"] = True
        return torch.tensor(0.0), {}

    monkeypatch.setattr("verl.trainer.ppo.score_centering.ppo_loss", fake_ppo_loss)
    data = _nested_batch([2], 3, 9)
    out = score_centering_ppo_loss(config, student_logits=torch.randn(1, 2, 9), data=data)
    assert "sc_correction" in out
    score_centering_ppo_loss(config, model_output={}, data=data)
    assert called["ppo"]
