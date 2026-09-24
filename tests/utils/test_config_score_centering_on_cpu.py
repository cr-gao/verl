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

import pytest
from omegaconf import OmegaConf

from verl.utils.config import _validate_score_centering_config


def _config(algorithm_sc=True, actor_sc=True, loss_mode="bypass_mode", topk_log_probs=128, use_v1=True):
    policy_loss = {"loss_mode": loss_mode}
    if actor_sc is not None:
        policy_loss["rollout_correction"] = {"score_centering": actor_sc}
    return OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {"score_centering": algorithm_sc}},
            "actor_rollout_ref": {
                "actor": {"policy_loss": policy_loss},
                "rollout": {"topk_log_probs": topk_log_probs},
            },
            "trainer": {"use_v1": use_v1},
        }
    )


def test_score_centering_config_accepts_consistent_settings():
    _validate_score_centering_config(_config())


def test_score_centering_disabled_skips_checks():
    _validate_score_centering_config(_config(algorithm_sc=False, actor_sc=None, loss_mode="vanilla", topk_log_probs=0))


@pytest.mark.parametrize(
    "algorithm_sc, actor_sc",
    [(True, None), (True, False), (False, True)],
)
def test_score_centering_requires_driver_and_actor_flags(algorithm_sc, actor_sc):
    with pytest.raises(ValueError, match="actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering"):
        _validate_score_centering_config(_config(algorithm_sc=algorithm_sc, actor_sc=actor_sc))


def test_score_centering_requires_bypass_mode_loss():
    with pytest.raises(ValueError, match="actor_rollout_ref.actor.policy_loss.loss_mode"):
        _validate_score_centering_config(_config(loss_mode="vanilla"))


def test_score_centering_requires_rollout_topk_log_probs():
    with pytest.raises(ValueError, match="actor_rollout_ref.rollout.topk_log_probs"):
        _validate_score_centering_config(_config(topk_log_probs=0))


@pytest.mark.parametrize("use_v1", [True, False])
def test_score_centering_accepts_both_trainers(use_v1):
    _validate_score_centering_config(_config(use_v1=use_v1))


def test_score_centering_tolerates_null_rollout_correction():
    config = _config(actor_sc=None)
    config.algorithm.rollout_correction = None
    config.actor_rollout_ref.actor.policy_loss.rollout_correction = None

    _validate_score_centering_config(config)
