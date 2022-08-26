from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING
from contextlib import asynccontextmanager

from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import numpy as np
    from grpc import aio
    from numpy.typing import NDArray  # pylint: disable=unused-import
    from google.protobuf.message import Message

    from bentoml import Service
    from bentoml.grpc.v1alpha1 import service_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_pb2_grpc as services
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, services = import_generated_stubs()
    exception_msg = (
        "'grpcio' is not installed. Please install it with 'pip install -U grpcio'"
    )
    aio = LazyLoader("aio", globals(), "grpc.aio", exc_msg=exception_msg)
    np = LazyLoader("np", globals(), "numpy")


def make_pb_ndarray(shape: tuple[int, ...]) -> pb.NDArray:
    arr: NDArray[np.float32] = t.cast("NDArray[np.float32]", np.random.rand(*shape))
    return pb.NDArray(
        shape=list(shape), dtype=pb.NDArray.DTYPE_FLOAT, float_values=arr.ravel()
    )


async def async_client_call(
    method: str,
    stub: services.BentoServiceStub,
    data: dict[str, Message],
    assert_data: pb.Response | t.Callable[[pb.Response], bool] | None = None,
    timeout: int | None = None,
) -> pb.Response:
    req = pb.Request(api_name=method, **data)
    output: pb.Response = await stub.Call(req, timeout=timeout)
    if assert_data:
        try:
            if callable(assert_data):
                assert assert_data(output)
            else:
                assert output == assert_data
        except AssertionError:
            raise AssertionError(f"Failed while checking data: {output}")

    return output


@asynccontextmanager
async def make_client(
    host_url: str,
) -> t.AsyncGenerator[services.BentoServiceStub, None]:
    async with aio.insecure_channel(host_url) as channel:
        # create a blocking call to wait til channel is ready.
        await channel.channel_ready()
        yield services.BentoServiceStub(channel)  # type: ignore (no generated stubs)


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
