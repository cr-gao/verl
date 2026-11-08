# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio

import pytest
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.agent_loop_tq as agent_loop_tq_module
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, _settle_session_tasks, apply_greedy_sampling_params
from verl.workers.rollout.tq_direct_write import TQ_SINK


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


_WorkerTQ = AgentLoopWorkerTQ.__ray_metadata__.modified_class


class _DummyWorker:
    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _compute_score = AgentLoopWorker._compute_score
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
    distillation_enabled = False

    def __init__(self, topk_log_probs: int):
        self.rollout_config = OmegaConf.create({"topk_log_probs": topk_log_probs})
        self.processor = None
        self.reward_loop_worker_handles = None


def _output(extra_fields: dict) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[11, 12],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


async def _postprocess(monkeypatch, worker: _DummyWorker, output: AgentLoopOutput, validate: bool):
    captured = {}

    async def async_kv_batch_put(*, keys, fields, tags, partition_id):
        captured["fields"] = fields

    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_batch_put", async_kv_batch_put)
    await _WorkerTQ._agent_loop_postprocess(
        worker, output, validate, uid="u0", session_id=0, global_steps=1, raw_prompt=[{"role": "user", "content": "hi"}]
    )
    return captured["fields"]


FIELDS = ["response_topk_ids", "response_topk_log_probs", "response_support_complete"]


def _worker_tq():
    return _WorkerTQ.__new__(_WorkerTQ)


def test_agent_loop_worker_tq_actor_class_is_serializable():
    # Ray exports the modified actor class by value, so every global its methods reach must pickle.
    import ray.cloudpickle as cloudpickle

    cloudpickle.dumps(_WorkerTQ)


@pytest.mark.asyncio
async def test_run_agent_loop_sets_sink_for_the_session_and_resets_it(monkeypatch):
    seen = {}

    async def parent(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        seen["sink"] = TQ_SINK.get()

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", parent)

    await _worker_tq()._run_agent_loop(
        {"topk_log_probs": 2}, {"validate": False}, agent_name="single_turn_agent", uid="u0", session_id=2
    )

    assert seen["sink"] == {"partition_id": "train", "key": "u0_2_0"}
    assert TQ_SINK.get() is None


@pytest.mark.asyncio
async def test_run_agent_loop_binds_no_sink_when_no_heads_are_requested(monkeypatch):
    seen = {}
    cleared = []

    async def parent(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        seen["sink"] = TQ_SINK.get()
        raise RuntimeError("boom")

    async def clear(*, keys, partition_id):
        cleared.append(keys)

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", parent)
    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_clear", clear)

    with pytest.raises(RuntimeError):
        await _worker_tq()._run_agent_loop({}, {"validate": False}, agent_name="x", uid="u0", session_id=0)

    assert seen["sink"] is None
    assert cleared == []


@pytest.mark.asyncio
async def test_run_agent_loop_clears_its_key_when_the_session_raises(monkeypatch):
    cleared = []

    async def parent(self, *args, **kwargs):
        raise RuntimeError("boom")

    async def clear(*, keys, partition_id):
        cleared.append((keys, partition_id))

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", parent)
    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_clear", clear)

    with pytest.raises(RuntimeError):
        await _worker_tq()._run_agent_loop(
            {"topk_log_probs": 2}, {"validate": True}, agent_name="x", uid="u0", session_id=0
        )

    assert cleared == [(["u0_0_0"], "val")]


@pytest.mark.asyncio
async def test_postprocess_strips_marker_and_emits_no_topk_columns(monkeypatch):
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), _output({"tq_fields": FIELDS}), False)

    assert "rollout_topk_ids" not in fields.keys() and "response_topk_ids" not in fields.keys()
    assert "tq_fields" not in fields["extra_fields"][0]


@pytest.mark.asyncio
async def test_postprocess_rejects_marker_on_multi_turn_session(monkeypatch):
    output = _output({"tq_fields": FIELDS})
    output.num_turns = 4

    with pytest.raises(ValueError, match="single-turn"):
        await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), output, False)


@pytest.mark.asyncio
async def test_postprocess_rejects_marker_on_multi_output_session(monkeypatch):
    outputs = [_output({"tq_fields": FIELDS}), _output({})]

    with pytest.raises(ValueError, match="single output"):
        await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), outputs, False)


@pytest.mark.asyncio
async def test_postprocess_clears_key_on_empty_output(monkeypatch):
    cleared = []

    async def clear(*, keys, partition_id):
        cleared.append((keys, partition_id))

    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_clear", clear)

    await _WorkerTQ._agent_loop_postprocess(
        _DummyWorker(topk_log_probs=2), [], False, uid="u0", session_id=0, global_steps=1, raw_prompt=[]
    )

    assert cleared == [(["u0_0_0"], "train")]


def test_greedy_reward_baseline_does_not_request_score_centering_heads():
    params = {"top_p": 0.9, "top_k": 64, "temperature": 1.0, "topk_log_probs": 128, "logprobs": True}
    apply_greedy_sampling_params(params)
    assert params == {"top_p": 1.0, "top_k": -1, "temperature": 0, "logprobs": True}


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_skips_rollout_topk_on_validate(monkeypatch):
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), _output({}), validate=True)

    assert "rollout_topk_ids" not in fields.keys()
    assert "rollout_topk_log_probs" not in fields.keys()
