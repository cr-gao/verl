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
import torch
from tensordict import TensorDict

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template


def test_padding_template_uses_dummy_heads():
    k = 3
    sample = TensorDict(
        {
            "prompts": torch.zeros(4, dtype=torch.int64),
            "responses": torch.zeros(2, dtype=torch.int64),
            "input_ids": torch.zeros(6, dtype=torch.int64),
            "attention_mask": torch.ones(6, dtype=torch.int64),
            "response_mask": torch.ones(2, dtype=torch.int64),
            "position_ids": torch.arange(6),
            "rollout_topk_ids": torch.zeros(6, k, dtype=torch.int32),
            "rollout_topk_log_probs": torch.zeros(6, k),
        },
        batch_size=[],
    )
    template, _ = construct_minimal_padding_template(sample, {}, eos_token_id=0)
    seq_len = template["input_ids"].shape[0]
    assert template["rollout_topk_ids"].shape == (seq_len, k)
    torch.testing.assert_close(template["rollout_topk_log_probs"].exp().sum(-1), torch.ones(seq_len))
