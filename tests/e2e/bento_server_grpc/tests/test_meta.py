from __future__ import annotations

import pytest

from bentoml.testing.grpc import create_channel
from bentoml.grpc.v1alpha1 import service_test_pb2 as pb_test
from bentoml.grpc.v1alpha1 import service_test_pb2_grpc as services_test


@pytest.mark.asyncio
async def test_success_invocation_custom_servicer(host: str) -> None:

    async with create_channel(host) as channel:
        stub = services_test.TestServiceStub(channel)  # type: ignore (no async types)
        request = pb_test.ExecuteRequest(input="BentoML")
        resp: pb_test.ExecuteResponse = await stub.Execute(request)
        assert resp.output == "Hello, BentoML!"
