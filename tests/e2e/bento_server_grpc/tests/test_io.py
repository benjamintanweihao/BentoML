from __future__ import annotations

import traceback
from typing import TYPE_CHECKING
from functools import partial

import pytest

from bentoml.testing.grpc import make_client
from bentoml.testing.grpc import make_pb_ndarray
from bentoml.testing.grpc import async_client_call
from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import grpc
    import numpy as np
    from grpc import aio
    from google.protobuf import wrappers_pb2

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs

    exception_msg = (
        "'grpcio' is not installed. Please install it with 'pip install -U grpcio'"
    )
    pb, _ = import_generated_stubs()
    grpc = LazyLoader("grpc", globals(), "grpc", exc_msg=exception_msg)
    aio = LazyLoader("aio", globals(), "grpc.aio", exc_msg=exception_msg)
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")
    np = LazyLoader("np", globals(), "numpy")


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
        await async_client_call(
            "double_ndarray",
            stub=client,
            data={"ndarray": make_pb_ndarray((1000,))},
            assert_data=partial(
                assert_ndarray, assert_shape=[1000], assert_dtype=pb.NDArray.DTYPE_FLOAT
            ),
        )
        await async_client_call(
            "double_ndarray",
            stub=client,
            data={"ndarray": pb.NDArray(shape=[2, 2], int32_values=[1, 2, 3, 4])},
            assert_data=lambda resp: resp.ndarray.int32_values == [2, 4, 6, 8],
        )
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "double_ndarray",
                stub=client,
                data={"ndarray": pb.NDArray(string_values=np.array(["2", "2f"]))},
            )
        assert context_exception.value.code() == grpc.StatusCode.INTERNAL
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "double_ndarray",
                stub=client,
                data={
                    "ndarray": pb.NDArray(
                        dtype=123, string_values=np.array(["2", "2f"])  # type: ignore (test exception)
                    )
                },
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "double_ndarray",
                stub=client,
                data={"raw_bytes_contents": np.array([1, 2, 3, 4]).ravel().tobytes()},
            )
        assert context_exception.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "double_ndarray",
                stub=client,
                data={"text": wrappers_pb2.StringValue(value="asdf")},
            )
        assert context_exception.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_ndarray_enforce_shape",
                stub=client,
                data={"ndarray": make_pb_ndarray((1000,))},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_ndarray_enforce_dtype",
                stub=client,
                data={"ndarray": pb.NDArray(string_values=np.array(["2", "2f"]))},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
