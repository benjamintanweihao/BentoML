from __future__ import annotations

import typing as t
import traceback
from typing import overload
from typing import TYPE_CHECKING
from contextlib import asynccontextmanager

from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import numpy as np
    from grpc import aio
    from numpy.typing import NDArray  # pylint: disable=unused-import
    from grpc_health.v1 import health_pb2_grpc as services_health
    from google.protobuf.message import Message

    from bentoml import Service
    from bentoml.grpc.v1alpha1 import service_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_pb2_grpc as services

    S = t.TypeVar("S", bound=t.Any)
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
    data: dict[str, Message | bytes | str | dict[str, t.Any]],
    assert_data: pb.Response | t.Callable[[pb.Response], bool] | None = None,
    timeout: int | None = None,
    sanity: bool = True,
) -> pb.Response:
    req = pb.Request(api_name=method, **data)
    output: pb.Response = await stub.Call(req, timeout=timeout)
    if sanity:
        assert output
    if assert_data:
        try:
            if callable(assert_data):
                assert assert_data(output)
            else:
                assert output == assert_data
        except AssertionError:
            raise AssertionError(f"Failed while checking data: {output}")
    return output


@overload
async def make_client(
    host_url: str, stubs: t.Type[services.BentoServiceStub]
) -> t.AsyncGenerator[services.BentoServiceStub, None]:
    ...


@overload
async def make_client(
    host_url: str, stubs: t.Type[services_health.HealthStub]
) -> t.AsyncGenerator[services_health.HealthStub, None]:
    ...


@asynccontextmanager
async def make_client(
    host_url: str, stubs: t.Type[S] = None
) -> t.AsyncGenerator[S, None]:
    if stubs is None:
        stubs = services.BentoServiceStub  # type: ignore (not yet supported)
    assert stubs
    try:
        async with aio.insecure_channel(host_url) as channel:
            # create a blocking call to wait til channel is ready.
            await channel.channel_ready()
            yield stubs(channel)
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
