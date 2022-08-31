from __future__ import annotations

import traceback
from typing import TYPE_CHECKING
from functools import partial

import pytest

from bentoml.testing.grpc import make_client
from bentoml.testing.grpc import make_pb_ndarray
from bentoml.testing.grpc import async_client_call

if TYPE_CHECKING:
    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()


def assert_ndarray(
    resp: pb.Response,
    assert_shape: list[int],
    assert_dtype: pb.NDArray.DType.ValueType,
) -> bool:

    dtype = resp.ndarray.dtype
    try:
        assert resp.ndarray.shape == assert_shape
        assert dtype == assert_dtype
        return True
    except AssertionError:
        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_numpy(host: str):
    async with make_client(host) as client:
        res = await async_client_call(
            "double_ndarray",
            stub=client,
            data={"ndarray": make_pb_ndarray((1000,))},
            assert_data=partial(
                assert_ndarray, assert_shape=[1000], assert_dtype=pb.NDArray.DTYPE_FLOAT
            ),
        )
        res = await async_client_call(
            "echo_ndarray_from_sample",
            stub=client,
            data={"ndarray": make_pb_ndarray((2, 2))},
        )

        assert res
