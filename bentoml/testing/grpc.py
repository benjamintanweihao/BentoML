from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING
from contextlib import asynccontextmanager

from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    from grpc import aio

    from bentoml import Service
    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.v1 import service_pb2_grpc as services
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, services = import_generated_stubs()
    exception_msg = (
        "'grpcio' is not installed. Please install it with 'pip install -U grpcio'"
    )
    aio = LazyLoader("aio", globals(), "grpc.aio", exc_msg=exception_msg)


@asynccontextmanager
async def create_stubs(
    host: str, port: int
) -> t.AsyncGenerator[services.BentoServiceStub, None]:
    async with aio.insecure_channel(f"{host}:{port}") as channel:
        yield services.BentoServiceStub(channel)  # type: ignore (no generated stubs)


@asynccontextmanager
async def create_server(
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
