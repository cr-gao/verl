# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.workers.rollout import tq_direct_write as mod

SINK = {"partition_id": "train", "key": "u1_0_0"}


def _capture(monkeypatch, existing=None):
    calls = {"init": 0, "put": [], "get": []}
    monkeypatch.setattr(mod, "_TQ_READY", False)
    monkeypatch.setattr(mod.tq, "init", lambda *a, **k: calls.__setitem__("init", calls["init"] + 1))

    async def put(*, keys, partition_id, fields, tags=None):
        calls["put"].append((keys, partition_id, fields))

    async def get(*, keys, partition_id, select_fields):
        calls["get"].append((keys, partition_id, select_fields))
        return existing

    monkeypatch.setattr(mod.tq, "async_kv_batch_put", put)
    monkeypatch.setattr(mod.tq, "async_kv_batch_get", get)
    return calls


@pytest.mark.asyncio
async def test_put_direct_fields_writes_jagged_columns_under_key(monkeypatch):
    calls = _capture(monkeypatch)
    ids = np.arange(6, dtype=np.int32).reshape(3, 2)
    lps = np.full((3, 2), -0.5, dtype=np.float32)
    fields = {"response_topk_ids": ids, "response_topk_log_probs": lps}

    written = await mod.put_direct_fields(SINK, fields)
    await mod.put_direct_fields(SINK, fields)

    assert written == ["response_topk_ids", "response_topk_log_probs"]
    keys, partition_id, stored = calls["put"][0]
    assert keys == ["u1_0_0"] and partition_id == "train" and stored.batch_size == torch.Size([1])
    assert stored["response_topk_ids"][0].tolist() == ids.tolist()
    assert stored["response_topk_ids"][0].dtype == torch.int32
    assert stored["response_topk_log_probs"][0].dtype == torch.float32
    assert calls["get"] == []
    assert calls["init"] == 1


@pytest.mark.asyncio
async def test_put_direct_fields_append_concatenates_existing_rows(monkeypatch):
    prev = TensorDict(
        {
            "response_topk_ids": torch.nested.as_nested_tensor(
                [torch.tensor([[1, 2]], dtype=torch.int32)], layout=torch.jagged
            ),
            "response_topk_log_probs": torch.nested.as_nested_tensor(
                [torch.tensor([[-1.0, -1.0]])], layout=torch.jagged
            ),
        },
        batch_size=[1],
    )
    calls = _capture(monkeypatch, existing=prev)

    await mod.put_direct_fields(
        {**SINK, "append": True},
        {
            "response_topk_ids": np.array([[3, 4]], dtype=np.int32),
            "response_topk_log_probs": np.array([[-2.0, -2.0]], dtype=np.float32),
        },
    )

    assert calls["get"] == [(["u1_0_0"], "train", ["response_topk_ids", "response_topk_log_probs"])]
    _, _, stored = calls["put"][0]
    assert stored["response_topk_ids"][0].tolist() == [[1, 2], [3, 4]]
    assert stored["response_topk_log_probs"][0].tolist() == [[-1.0, -1.0], [-2.0, -2.0]]
