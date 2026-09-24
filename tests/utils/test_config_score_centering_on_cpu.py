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

_RC = {"bypass_mode": True, "loss_type": "reinforce", "rollout_is": "token", "rollout_is_threshold": 2.0}


def _config(
    algorithm_sc=True,
    actor_sc=True,
    loss_mode="bypass_mode",
    topk_log_probs=128,
    use_v1=True,
    algorithm_rc=None,
    actor_rc=None,
):
    policy_loss = {"loss_mode": loss_mode}
    if actor_sc is not None:
        policy_loss["rollout_correction"] = {**_RC, **(actor_rc or {}), "score_centering": actor_sc}
    return OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {**_RC, **(algorithm_rc or {}), "score_centering": algorithm_sc}},
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


def test_score_centering_accepts_actor_side_without_is_keys():
    config = _config()
    policy_loss = config.actor_rollout_ref.actor.policy_loss
    policy_loss.rollout_correction = {"bypass_mode": True, "loss_type": "reinforce", "score_centering": True}
    config.algorithm.rollout_correction.rollout_is = None

    _validate_score_centering_config(config)


@pytest.mark.parametrize(
    "actor_rc, match",
    [
        ({"rollout_is": "sequence"}, "actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is"),
        ({"rollout_is_batch_normalize": True}, "rollout_correction.rollout_is_batch_normalize"),
    ],
)
def test_score_centering_rejects_actor_side_is_settings(actor_rc, match):
    with pytest.raises(ValueError, match=match):
        _validate_score_centering_config(_config(actor_rc=actor_rc))


@pytest.mark.parametrize(
    "key, actor_value",
    [
        ("bypass_mode", False),
        ("loss_type", "ppo_clip"),
        ("rollout_is", None),
        ("rollout_is_threshold", 3.0),
    ],
)
def test_score_centering_requires_matching_actor_side(key, actor_value):
    with pytest.raises(ValueError, match=f"rollout_correction.{key}"):
        _validate_score_centering_config(_config(actor_rc={key: actor_value}))


def test_score_centering_reads_missing_actor_side_rollout_is_as_none():
    config = _config()
    del config.actor_rollout_ref.actor.policy_loss.rollout_correction["rollout_is"]

    with pytest.raises(ValueError, match=r"actor_rollout_ref\.actor\.policy_loss\.rollout_correction\.rollout_is "):
        _validate_score_centering_config(config)
