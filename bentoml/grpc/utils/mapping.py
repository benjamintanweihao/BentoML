"""
Static mapping from BentoML protobuf message to values.

For all function in this module, make sure to lazy load the generated protobuf.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from functools import lru_cache

from bentoml._internal.utils.lazy_loader import LazyLoader as _LazyLoader

if TYPE_CHECKING:
    from enum import Enum

    import grpc

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils._import_hook import import_generated_stubs

    grpc = _LazyLoader(
        "grpc",
        globals(),
        "grpc",
        exc_msg="'grpc' is required. Install with 'pip install grpcio'.",
    )
    pb, _ = import_generated_stubs()

    del _LazyLoader


@lru_cache(maxsize=1)
def http_status_to_grpc_status_map() -> dict[Enum, grpc.StatusCode]:
    # Maps HTTP status code to grpc.StatusCode
    from http import HTTPStatus

    return {
        HTTPStatus.OK: grpc.StatusCode.OK,
        HTTPStatus.UNAUTHORIZED: grpc.StatusCode.UNAUTHENTICATED,
        HTTPStatus.FORBIDDEN: grpc.StatusCode.PERMISSION_DENIED,
        HTTPStatus.NOT_FOUND: grpc.StatusCode.UNIMPLEMENTED,
        HTTPStatus.TOO_MANY_REQUESTS: grpc.StatusCode.UNAVAILABLE,
        HTTPStatus.BAD_GATEWAY: grpc.StatusCode.UNAVAILABLE,
        HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
        HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
        HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
        HTTPStatus.UNPROCESSABLE_ENTITY: grpc.StatusCode.FAILED_PRECONDITION,
    }


@lru_cache(maxsize=1)
def grpc_status_to_http_status_map() -> dict[grpc.StatusCode, Enum]:
    return {v: k for k, v in http_status_to_grpc_status_map().items()}


@lru_cache(maxsize=1)
def filetype_pb_to_mimetype_map() -> dict[pb.File.FileType.ValueType, str]:
    return {
        pb.File.FILE_TYPE_CSV: "text/csv",
        pb.File.FILE_TYPE_PLAINTEXT: "text/plain",
        pb.File.FILE_TYPE_JSON: "application/json",
        pb.File.FILE_TYPE_BYTES: "application/octet-stream",
        pb.File.FILE_TYPE_PDF: "application/pdf",
        pb.File.FILE_TYPE_PNG: "image/png",
        pb.File.FILE_TYPE_JPEG: "image/jpeg",
        pb.File.FILE_TYPE_GIF: "image/gif",
        pb.File.FILE_TYPE_TIFF: "image/tiff",
        pb.File.FILE_TYPE_BMP: "image/bmp",
        pb.File.FILE_TYPE_WEBP: "image/webp",
        pb.File.FILE_TYPE_SVG: "image/svg+xml",
    }


@lru_cache(maxsize=1)
def mimetype_to_filetype_pb_map() -> dict[str, pb.File.FileType.ValueType]:
    return {v: k for k, v in filetype_pb_to_mimetype_map().items()}
