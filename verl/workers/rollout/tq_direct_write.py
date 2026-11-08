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
"""Write per-sample fields from a rollout server straight into TransferQueue.

A sink names the TransferQueue partition and key of the sample being generated. Fields
that are final on the server side are written under that key here, so they never travel
back through the agent loop.
"""

from contextvars import ContextVar

import numpy as np
import torch

from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.transferqueue_utils import tq

TQ_SINK: ContextVar[dict | None] = ContextVar("tq_sink", default=None)

_TQ_READY = False


async def put_direct_fields(sink: dict, fields: dict[str, np.ndarray | torch.Tensor]) -> list[str]:
    """Store ``fields`` as one-sample jagged columns under ``sink["key"]``.

    With ``sink["append"]`` the rows are concatenated after the rows already stored for
    the same names. Returns the names written.
    """
    global _TQ_READY
    if not _TQ_READY:
        tq.init()
        _TQ_READY = True

    names = list(fields)
    tensors = {name: torch.as_tensor(value) for name, value in fields.items()}
    if sink.get("append"):
        existing = await tq.async_kv_batch_get(
            keys=[sink["key"]], partition_id=sink["partition_id"], select_fields=names
        )
        tensors = {name: torch.cat([existing[name][0], tensors[name]], dim=0) for name in names}

    await tq.async_kv_batch_put(
        keys=[sink["key"]], partition_id=sink["partition_id"], fields=list_of_dict_to_tensordict([tensors])
    )
    return names


async def emit_direct_fields(
    sink: dict | None, extra_fields: dict, fields: dict[str, np.ndarray | torch.Tensor]
) -> None:
    """Hand ``fields`` to the sink and leave only their names in ``extra_fields``.

    Without a sink the arrays themselves go into ``extra_fields``.
    """
    if sink is None:
        extra_fields.update(fields)
        return
    extra_fields["tq_fields"] = await put_direct_fields(sink, fields)
