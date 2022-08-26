from __future__ import annotations

import typing as t
import functools
from typing import TYPE_CHECKING
from dataclasses import asdict
from dataclasses import dataclass

import grpc
from grpc import aio

if TYPE_CHECKING:
    from bentoml.grpc.types import Request
    from bentoml.grpc.types import Response
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import HandlerCallDetails
    from bentoml.grpc.types import BentoServicerContext

    HandlerMethod = t.Callable[[Request, grpc.RpcContext], Response]


@dataclass
class Context:
    usage: str
    accuracy_score: float


class ContextInterceptor(grpc.ServerInterceptor):
    def __init__(self, *, usage: str, accuracy_score: float) -> None:
        self.context = Context(usage=usage, accuracy_score=accuracy_score)
        self._record: set[str] = set()

    def intercept_service(
        self,
        continuation: t.Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ):
        from bentoml.grpc.utils import wrap_rpc_handler

        def wrapper(
            behaviour: HandlerMethod,
            request_streaming: bool,
            response_streaming: bool,
        ):
            def new_behaviour(
                request_or_iterator: Request, context: grpc.RpcContext
            ) -> Response:
                if response_streaming or request_streaming:
                    return behaviour(request_or_iterator, context)

                self._record.update(
                    {f"{self.context.usage}:{self.context.accuracy_score}"}
                )
                return behaviour(request_or_iterator, context)

            return new_behaviour

        return wrap_rpc_handler(wrapper, continuation(handler_call_details))


class AsyncContextInterceptor(aio.ServerInterceptor):
    def __init__(self, **kwargs: t.Any):
        self._sync = ContextInterceptor(**kwargs)

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        from bentoml.grpc.utils import wrap_rpc_handler

        handler = await continuation(handler_call_details)
        method_name = handler_call_details.method

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        return wrap_rpc_handler(
            functools.partial(self._sync_wrapper, method_name), handler
        )

    def _sync_wrapper(self, method_name: str, prev_handler: RpcMethodHandler):
        async def new_handler(
            request_or_iterator: Request, context: BentoServicerContext
        ) -> Response:
            assert method_name == "Call"
            response: Response = await prev_handler(request_or_iterator, context)
            context.set_trailing_metadata(
                aio.Metadata.from_tuple(
                    tuple([(k, v) for k, v in asdict(self._sync.context).items()])
                )
            )
            return response

        return new_handler
