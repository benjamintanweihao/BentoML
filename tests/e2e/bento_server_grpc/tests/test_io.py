from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bentoml.testing.grpc import make_client
from bentoml.testing.grpc import make_pb_ndarray
from bentoml.testing.grpc import async_client_call

if TYPE_CHECKING:
    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()


@pytest.mark.asyncio
async def test_numpy(host: str):
    async with make_client(host) as client:
        res = await async_client_call(
            "double_ndarray",
            stub=client,
            data={"ndarray": make_pb_ndarray((1000,))},
            assert_data=lambda resp: resp.ndarray.dtype
            == pb.NDArray.DTYPE_FLOAT.as_integer_ratio()[1]
            and resp.ndarray.shape == [1000],
        )
        res = await async_client_call(
            "echo_ndarray_from_sample",
            stub=client,
            data={"ndarray": make_pb_ndarray((2, 2))},
        )

        assert res
