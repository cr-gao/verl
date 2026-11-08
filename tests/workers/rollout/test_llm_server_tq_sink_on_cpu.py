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

from types import SimpleNamespace

import pytest

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.tq_direct_write import TQ_SINK

SINK = {"partition_id": "train", "key": "u_0_0"}


def _client_with_fake_server(monkeypatch):
    seen = {}

    async def remote(**kwargs):
        seen["generate"] = kwargs
        return TokenOutput(token_ids=[1], stop_reason="completed")

    fake_server = SimpleNamespace(generate=SimpleNamespace(remote=remote))

    async def acquire(self, request_id, **kwargs):
        seen["acquire"] = kwargs
        return "s0", fake_server

    monkeypatch.setattr(LLMServerClient, "_acquire_server", acquire)
    monkeypatch.setattr(LLMServerClient, "_release_server", lambda self, server_id, request_id: None)
    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="vllm")))
    return LLMServerClient(config=config, load_balancer_handle=None), seen


@pytest.mark.asyncio
async def test_client_forwards_context_sink_as_explicit_kwarg(monkeypatch):
    client, seen = _client_with_fake_server(monkeypatch)
    token = TQ_SINK.set(SINK)
    try:
        await client.generate(request_id="r", prompt_ids=[1], sampling_params={})
    finally:
        TQ_SINK.reset(token)

    assert seen["generate"]["tq_sink"] == SINK
    assert "tq_sink" not in seen["acquire"]


@pytest.mark.asyncio
async def test_client_passes_no_sink_when_context_unset(monkeypatch):
    client, seen = _client_with_fake_server(monkeypatch)

    await client.generate(request_id="r", prompt_ids=[1], sampling_params={})

    assert "tq_sink" not in seen["generate"]
