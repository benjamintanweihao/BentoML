from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import grpc
    from grpc import aio

    from bentoml.grpc.types import Request
    from bentoml.grpc.types import BentoUnaryUnaryCall
else:
    aio = LazyLoader("aio", globals(), "grpc.aio")


class AssertClientInterceptor(aio.UnaryUnaryClientInterceptor):
    def __init__(
        self,
        assert_code: grpc.StatusCode | None = None,
        assert_details: str | None = None,
    ):
        self._assert_code = assert_code
        self._assert_details = assert_details

    async def intercept_unary_unary(  # type: ignore (unable to infer types from parameters)
        self,
        continuation: t.Callable[[aio.ClientCallDetails, Request], BentoUnaryUnaryCall],
        client_call_details: aio.ClientCallDetails,
        request: Request,
    ) -> BentoUnaryUnaryCall:
        # Note that we cast twice here since grpc.aio._call.UnaryUnaryCall
        # implements __await__, which returns ResponseType. However, pyright
        # are unable to determine types from given mixin.
        #
        # continuation(client_call_details, request) -> call: UnaryUnaryCall
        # await call -> ResponseType
        call = await t.cast(
            "t.Awaitable[BentoUnaryUnaryCall]",
            continuation(client_call_details, request),
        )
        if self._assert_code:
            assert await call.code() == self._assert_code
        if self._assert_details:
            assert self._assert_details in await call.details()
        return call
