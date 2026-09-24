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

"""CPU coverage for the score centering config surface."""

import pytest

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.workers.config.rollout import RolloutConfig


def test_score_centering_presets_are_bypass_reinforce():
    for cfg in (
        RolloutCorrectionConfig.bypass_pg_sc(),
        RolloutCorrectionConfig.bypass_pg_token_tis_sc(),
        RolloutCorrectionConfig.bypass_pg_token_icepop_sc(),
    ):
        assert cfg.score_centering and cfg.bypass_mode and cfg.loss_type == "reinforce"
    assert RolloutCorrectionConfig.bypass_pg_sc().rollout_is is None
    assert RolloutCorrectionConfig.bypass_pg_token_tis_sc(threshold=3.0).rollout_is_threshold == 3.0
    assert RolloutCorrectionConfig.bypass_pg_token_icepop_sc().rollout_is_threshold == "0.5_5.0"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bypass_mode=False, loss_type="reinforce"),
        dict(bypass_mode=True, loss_type="ppo_clip"),
        dict(bypass_mode=True, loss_type="reinforce", rollout_is="sequence"),
        dict(bypass_mode=True, loss_type="reinforce", rollout_is="token", rollout_is_batch_normalize=True),
    ],
)
def test_score_centering_rejects_unsupported_modes(kwargs):
    with pytest.raises(ValueError, match="score_centering"):
        RolloutCorrectionConfig(score_centering=True, **kwargs)


def test_topk_log_probs_requires_calculate_log_probs_and_full_support():
    with pytest.raises(ValueError, match="calculate_log_probs"):
        RolloutConfig(topk_log_probs=128, calculate_log_probs=False)
    with pytest.raises(ValueError, match="top_p"):
        RolloutConfig(topk_log_probs=128, calculate_log_probs=True, top_p=0.9)
    with pytest.raises(ValueError, match="top_k"):
        RolloutConfig(topk_log_probs=128, calculate_log_probs=True, top_k=50)


def test_topk_log_probs_requires_processed_logprobs():
    with pytest.raises(ValueError, match="processed_logprobs"):
        RolloutConfig(name="vllm", topk_log_probs=128, calculate_log_probs=True, logprobs_mode="raw_logprobs")


def test_topk_log_probs_rejects_negative_head_size():
    with pytest.raises(ValueError, match="must be >= 0"):
        RolloutConfig(name="vllm", topk_log_probs=-1, calculate_log_probs=True)


def test_topk_log_probs_requires_vllm_rollout():
    with pytest.raises(ValueError, match="vLLM"):
        RolloutConfig(name="sglang", topk_log_probs=128, calculate_log_probs=True)


def test_topk_log_probs_raises_vllm_max_logprobs():
    cfg = RolloutConfig(name="vllm", topk_log_probs=128, calculate_log_probs=True)
    assert cfg.engine_kwargs["vllm"]["max_logprobs"] == 128
    cfg = RolloutConfig(
        name="vllm", topk_log_probs=32, calculate_log_probs=True, engine_kwargs={"vllm": {"max_logprobs": 64}}
    )
    assert cfg.engine_kwargs["vllm"]["max_logprobs"] == 64
    with pytest.raises(ValueError, match="max_logprobs"):
        RolloutConfig(
            name="vllm", topk_log_probs=128, calculate_log_probs=True, engine_kwargs={"vllm": {"max_logprobs": 20}}
        )
