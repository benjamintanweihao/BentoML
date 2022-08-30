from __future__ import annotations

import os
import typing as t
from typing import TYPE_CHECKING

import psutil
import pytest

if TYPE_CHECKING:
    from contextlib import ExitStack

    from _pytest.fixtures import FixtureRequest as _PytestFixtureRequest

    class FixtureRequest(_PytestFixtureRequest):
        param: str


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(params=["default.yml", "cors_enabled.yml"], scope="session")
def server_config_file(request: FixtureRequest) -> str:
    return os.path.join(PROJECT_DIR, "configs", request.param)


@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: str,
    server_config_file: str,
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
        config_file=server_config_file,
        deployment_mode=deployment_mode,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
    ) as host:
        yield host
