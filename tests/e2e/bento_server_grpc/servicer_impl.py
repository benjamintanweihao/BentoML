# pylint: disable=unused-argument
from __future__ import annotations

from typing import TYPE_CHECKING

from bentoml.grpc.v1alpha1 import service_test_pb2 as pb
from bentoml.grpc.v1alpha1 import service_test_pb2_grpc as services

if TYPE_CHECKING:
    from grpc import aio


class TestServiceServicer(services.TestServiceServicer):
    def Execute(
        self,
        request: pb.ExecuteRequest,
        context: aio.ServicerContext[pb.ExecuteRequest, pb.ExecuteResponse],
    ) -> pb.ExecuteResponse:
        return pb.ExecuteResponse(output="Hello, {}!".format(request.input))
