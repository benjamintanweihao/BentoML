from __future__ import annotations

import pytest

from bentoml.testing.grpc import make_client
from bentoml.testing.grpc import make_pb_ndarray
from bentoml.testing.grpc import async_client_call


@pytest.mark.asyncio
async def test_numpy(host: str):
    async with make_client(host) as client:
        array = make_pb_ndarray((2, 2))

        res = await async_client_call(
            "echo_ndarray_from_sample", client, {"ndarray": array}
        )

        assert res
