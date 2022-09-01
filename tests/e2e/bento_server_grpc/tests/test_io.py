from __future__ import annotations

import io
import random
import typing as t
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
    import pandas as pd
    import numpy as np
    import PIL.Image as PILImage
    from grpc import aio
    from google.protobuf import struct_pb2
    from google.protobuf import wrappers_pb2

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_pb2_grpc as services
else:
    from bentoml.grpc.utils import import_generated_stubs

    exception_msg = (
        "'grpcio' is not installed. Please install it with 'pip install -U grpcio'"
    )
    pb, services = import_generated_stubs()
    grpc = LazyLoader("grpc", globals(), "grpc", exc_msg=exception_msg)
    aio = LazyLoader("aio", globals(), "grpc.aio", exc_msg=exception_msg)
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")
    struct_pb2 = LazyLoader("struct_pb2", globals(), "google.protobuf.struct_pb2")
    np = LazyLoader("np", globals(), "numpy")
    pd = LazyLoader("pd", globals(), "pandas")
    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")


def assert_ndarray(
    resp: pb.Response,
    assert_shape: list[int],
    assert_dtype: pb.NDArray.DType.ValueType,
) -> bool:
    __tracebackhide__ = True  # Hide traceback for py.test

    dtype = resp.ndarray.dtype
    try:
        assert resp.ndarray.shape == assert_shape
        assert dtype == assert_dtype
        return True
    except AssertionError:
        traceback.print_exc()
        return False


def make_iris_proto(**fields: struct_pb2.Value) -> struct_pb2.Value:
    return struct_pb2.Value(
        struct_value=struct_pb2.Struct(
            fields={
                "request_id": struct_pb2.Value(string_value="123"),
                "iris_features": struct_pb2.Value(
                    struct_value=struct_pb2.Struct(fields=fields)
                ),
            }
        )
    )


@pytest.mark.asyncio
async def test_numpy(host: str):
    async with make_client(host) as client_stub:  # type: ignore (no infer types)
        if TYPE_CHECKING:
            client_stub = t.cast(services.BentoServiceStub, client_stub)
        await async_client_call(
            "double_ndarray",
            stub=client_stub,
            data={"ndarray": make_pb_ndarray((1000,))},
            assert_data=partial(
                assert_ndarray, assert_shape=[1000], assert_dtype=pb.NDArray.DTYPE_FLOAT
            ),
        )
        await async_client_call(
            "double_ndarray",
            stub=client_stub,
            data={"ndarray": pb.NDArray(shape=[2, 2], int32_values=[1, 2, 3, 4])},
            assert_data=lambda resp: resp.ndarray.int32_values == [2, 4, 6, 8],
        )
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "double_ndarray",
                stub=client_stub,
                data={"ndarray": pb.NDArray(string_values=np.array(["2", "2f"]))},
            )
        assert context_exception.value.code() == grpc.StatusCode.INTERNAL
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "double_ndarray",
                stub=client_stub,
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
                stub=client_stub,
                data={"raw_bytes_contents": np.array([1, 2, 3, 4]).ravel().tobytes()},
            )
        assert context_exception.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "double_ndarray",
                stub=client_stub,
                data={"text": wrappers_pb2.StringValue(value="asdf")},
            )
        assert context_exception.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_ndarray_enforce_shape",
                stub=client_stub,
                data={"ndarray": make_pb_ndarray((1000,))},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_ndarray_enforce_dtype",
                stub=client_stub,
                data={"ndarray": pb.NDArray(string_values=np.array(["2", "2f"]))},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_json(host: str):
    async with make_client(host) as client_stub:  # type: ignore (no infer types)
        if TYPE_CHECKING:
            client_stub = t.cast(services.BentoServiceStub, client_stub)
        await async_client_call(
            "echo_json",
            stub=client_stub,
            data={"json": struct_pb2.Value(string_value='"hi"')},
            assert_data=lambda resp: resp.json.string_value == '"hi"',
        )
        await async_client_call(
            "echo_json",
            stub=client_stub,
            data={
                "raw_bytes_contents": b'{"request_id": "123", "iris_features": {"sepal_len":2.34,"sepal_width":1.58, "petal_len":6.52, "petal_width":3.23}}'
            },
            assert_data=lambda resp: resp.json  # type: ignore (bad lambda types)
            == make_iris_proto(
                sepal_len=struct_pb2.Value(number_value=2.34),
                sepal_width=struct_pb2.Value(number_value=1.58),
                petal_len=struct_pb2.Value(number_value=6.52),
                petal_width=struct_pb2.Value(number_value=3.23),
            ),
        )
        await async_client_call(
            "echo_json_validate",
            stub=client_stub,
            data={
                "json": make_iris_proto(
                    **{
                        k: struct_pb2.Value(number_value=random.uniform(1.0, 6.0))
                        for k in [
                            "sepal_len",
                            "sepal_width",
                            "petal_len",
                            "petal_width",
                        ]
                    }
                )
            },
        )
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_json",
                stub=client_stub,
                data={"raw_bytes_contents": b"\n?xfa"},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_json",
                stub=client_stub,
                data={"text": wrappers_pb2.StringValue(value="asdf")},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_json_validate",
                stub=client_stub,
                data={
                    "json": make_iris_proto(
                        sepal_len=struct_pb2.Value(number_value=2.34),
                        sepal_width=struct_pb2.Value(number_value=1.58),
                        petal_len=struct_pb2.Value(number_value=6.52),
                    ),
                },
            )
        assert context_exception.value.code() == grpc.StatusCode.FAILED_PRECONDITION


@pytest.mark.asyncio
async def test_file(host: str, bin_file: bytes):
    # Test File as binary
    with open(str(bin_file), "rb") as f:
        fb = f.read()

    async with make_client(host) as client_stub:  # type: ignore (no infer types)
        if TYPE_CHECKING:
            client_stub = t.cast(services.BentoServiceStub, client_stub)
        await async_client_call(
            "predict_file",
            stub=client_stub,
            data={"raw_bytes_contents": fb},
            assert_data=lambda resp: resp.file.content == fb,
        )
        await async_client_call(
            "predict_file",
            stub=client_stub,
            data={"file": pb.File(kind=pb.File.FILE_TYPE_BYTES, content=fb)},
            assert_data=lambda resp: resp.file.content == b"\x810\x899"
            and resp.file.kind == pb.File.FILE_TYPE_BYTES,
        )
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "predict_file",
                stub=client_stub,
                data={"file": pb.File(kind=123, content=fb)},  # type: ignore (testing exception)
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "predict_file",
                stub=client_stub,
                data={"file": pb.File(kind=pb.File.FILE_TYPE_PDF, content=fb)},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "predict_file",
                stub=client_stub,
                data={"text": wrappers_pb2.StringValue(value="asdf")},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def assert_image(
    resp: pb.Response, assert_kind: pb.File.FileType.ValueType, im_file: bytes
) -> bool:
    __tracebackhide__ = True  # Hide traceback for py.test

    fio = io.BytesIO(resp.file.content)
    fio.name = "test.bmp"
    img = PILImage.open(fio)
    a1 = np.array(img)
    a2 = PILImage.open(im_file)

    assert resp.file.kind == assert_kind

    try:
        np.testing.assert_array_almost_equal(a1, np.array(a2))
        return True
    except AssertionError:
        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_image(host: str, img_file: bytes):
    # Test File as binary
    with open(str(img_file), "rb") as f:
        fb = f.read()

    async with make_client(host) as client_stub:  # type: ignore (no infer types)
        if TYPE_CHECKING:
            client_stub = t.cast(services.BentoServiceStub, client_stub)
        await async_client_call(
            "echo_image",
            stub=client_stub,
            data={"raw_bytes_contents": fb},
            assert_data=partial(
                assert_image, im_file=img_file, assert_kind=pb.File.FILE_TYPE_BMP
            ),
        )
        await async_client_call(
            "echo_image",
            stub=client_stub,
            data={"file": pb.File(kind=pb.File.FILE_TYPE_BMP, content=fb)},
            assert_data=partial(
                assert_image, im_file=img_file, assert_kind=pb.File.FILE_TYPE_BMP
            ),
        )
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_image",
                stub=client_stub,
                data={"file": pb.File(kind=123, content=fb)},  # type: ignore (testing exception)
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_image",
                stub=client_stub,
                data={"file": pb.File(kind=pb.File.FILE_TYPE_PDF, content=fb)},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(aio.AioRpcError) as context_exception:
            await async_client_call(
                "echo_image",
                stub=client_stub,
                data={"text": wrappers_pb2.StringValue(value="asdf")},
            )
        assert context_exception.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_pandas(host: str):
    ...
