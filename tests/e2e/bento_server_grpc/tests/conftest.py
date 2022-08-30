# pylint: disable=unused-argument
from __future__ import annotations

import os
import typing as t
from typing import TYPE_CHECKING

import psutil
import pytest

if TYPE_CHECKING:
    from contextlib import ExitStack


@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: str,
    clean_context: ExitStack,
) -> t.Generator[str, None, None]:
    if (
        (psutil.WINDOWS or psutil.MACOS)
        and os.environ.get("GITHUB_ACTIONS")
        and deployment_mode == "docker"
    ):
        pytest.skip(
            "Due to GitHub Action's licensing limitation, Docker deployment tests are not running on Windows/MacOS. Note that this test can still be run locally on Windows/MacOS if Docker is present."
        )

    if not psutil.LINUX and deployment_mode == "distributed":
        pytest.skip("Distributed deployment tests are only supported on Linux.")

    from bentoml.testing.server import host_bento

    with host_bento(
        "service:svc",
        deployment_mode=deployment_mode,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
        grpc=True,
    ) as host:
        yield host
