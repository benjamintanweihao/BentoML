from __future__ import annotations

import pytest

# @pytest.mark.asyncio
# async def test_numpy(host: str):
#     from bentoml.testing.grpc import make_client
#     from bentoml.testing.grpc import make_pb_ndarray
#     from bentoml.testing.grpc import async_client_call

#     async with make_client(host) as test_client:
#         array = make_pb_ndarray((2, 2))

#         res = await async_client_call("double_ndarray", test_client, {"ndarray": array})

#         assert res
