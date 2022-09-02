from __future__ import annotations

import typing as t
import traceback
from typing import TYPE_CHECKING
from contextlib import asynccontextmanager

from bentoml._internal.utils import LazyLoader
from bentoml.grpc.interceptors.client import AssertClientInterceptor

if TYPE_CHECKING:
    import grpc
    import numpy as np
    from grpc import aio
    from numpy.typing import NDArray  # pylint: disable=unused-import
    from grpc.aio._channel import Channel
    from google.protobuf.message import Message

    from bentoml import Service
    from bentoml.grpc.types import BentoUnaryUnaryCall
    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()
    exception_msg = (
        "'grpcio' is not installed. Please install it with 'pip install -U grpcio'"
    )
    grpc = LazyLoader("grpc", globals(), "grpc", exc_msg=exception_msg)
    aio = LazyLoader("aio", globals(), "grpc.aio", exc_msg=exception_msg)
    np = LazyLoader("np", globals(), "numpy")


def make_pb_ndarray(shape: tuple[int, ...]) -> pb.NDArray:
    arr: NDArray[np.float32] = t.cast("NDArray[np.float32]", np.random.rand(*shape))
    return pb.NDArray(
        shape=list(shape), dtype=pb.NDArray.DTYPE_FLOAT, float_values=arr.ravel()
    )


async def async_client_call(
    method: str,
    channel: Channel,
    data: dict[str, Message | bytes | str | dict[str, t.Any]],
    assert_data: pb.Response | t.Callable[[pb.Response], bool] | None = None,
    assert_code: grpc.StatusCode | None = None,
    assert_details: str | None = None,
    timeout: int | None = None,
    sanity: bool = True,
) -> pb.Response:
    if assert_code is None:
        # by default, we want to check if the request is healthy
        assert_code = grpc.StatusCode.OK
    try:
        # we will handle adding our testing interceptors here.
        # note that we shouldn't use private imports, but this will do
        channel._unary_unary_interceptors.append(  # type: ignore (private warning)
            AssertClientInterceptor(
                assert_code=assert_code, assert_details=assert_details
            )
        )
        Call = channel.unary_unary(
            "/bentoml.grpc.v1alpha1.BentoService/Call",
            request_serializer=pb.Request.SerializeToString,
            response_deserializer=pb.Response.FromString,
        )
        output = await t.cast(
            t.Awaitable[pb.Response],
            Call(pb.Request(api_name=method, **data), timeout=timeout),
        )
        if sanity:
            assert output
        if assert_data:
            try:
                if callable(assert_data):
                    assert assert_data(output)
                else:
                    assert output == assert_data
            except AssertionError:
                raise AssertionError(
                    f"Failed while checking data: {output.SerializeToString()}"
                )
        return output
    finally:
        # we will reset interceptors per call
        channel._unary_unary_interceptors = []  # type: ignore (private warning)


@asynccontextmanager
async def create_channel(host_url: str) -> t.AsyncGenerator[Channel, None]:
    try:
        async with aio.insecure_channel(host_url) as channel:
            # create a blocking call to wait til channel is ready.
            await channel.channel_ready()
            yield channel
        await channel.close()
    except aio.AioRpcError as e:
        traceback.print_exc()
        raise e from None


async def make_standalone_server(
    service: Service, bind_address: str
) -> t.AsyncGenerator[aio.Server, None]:
    from bentoml._internal.server import grpc as grpc_server

    config = grpc_server.Config(
        grpc_server.Servicer(service), bind_address=bind_address, enable_reflection=True
    )
    svr = grpc_server.Server(config).load()
    assert svr.loaded

    await svr.startup()

    yield svr.server

    await svr.shutdown()
