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

import inspect

import pytest

pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer  # noqa: E402


def test_generate_accepts_tq_sink():
    assert "tq_sink" in inspect.signature(vLLMHttpServer.generate).parameters


def test_generate_hands_topk_heads_to_the_direct_write_helper():
    source = inspect.getsource(vLLMHttpServer.generate)
    assert "emit_direct_fields(" in source
    assert 'extra_fields["response_topk_ids"]' not in source


def test_pd_forwards_the_sink_to_the_decode_leg_only():
    source = inspect.getsource(vLLMHttpServer._pd_dispatch)
    prefill_call = source.split("prefill_out = await self.generate(")[1].split("\n        )")[0]
    decode_call = source.split("return await decode_peer.generate.remote(")[1].split("\n        )")[0]
    assert "tq_sink" not in prefill_call
    assert "tq_sink=tq_sink" in decode_call
    assert "tq_sink" in inspect.signature(vLLMHttpServer._pd_dispatch).parameters
