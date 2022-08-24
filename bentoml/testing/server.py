# pylint: disable=redefined-outer-name,not-context-manager
from __future__ import annotations

import os
import sys
import time
import socket
import typing as t
import urllib
import logging
import itertools
import contextlib
import subprocess
import urllib.error
import urllib.request
import multiprocessing
from typing import TYPE_CHECKING
from contextlib import contextmanager

from bentoml._internal.tag import Tag
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils import cached_contextmanager
from bentoml._internal.utils.platform import kill_subprocess_tree
from bentoml._internal.configuration.containers import BentoMLContainer

logger = logging.getLogger("bentoml")


if TYPE_CHECKING:
    from aiohttp.typedefs import LooseHeaders
    from starlette.datastructures import Headers
    from starlette.datastructures import FormData

    from bentoml._internal.bento.bento import Bento


async def parse_multipart_form(headers: "Headers", body: bytes) -> "FormData":
    """
    parse starlette forms from headers and body
    """

    from starlette.formparsers import MultiPartParser

    async def async_bytesio(bytes_: bytes) -> t.AsyncGenerator[bytes, None]:
        yield bytes_
        yield b""
        return

    parser = MultiPartParser(headers=headers, stream=async_bytesio(body))
    return await parser.parse()


async def async_request(
    method: str,
    url: str,
    headers: t.Optional["LooseHeaders"] = None,
    data: t.Any = None,
    timeout: t.Optional[int] = None,
) -> t.Tuple[int, "Headers", bytes]:
    """
    A HTTP client with async API.
    """
    import aiohttp
    from starlette.datastructures import Headers

    async with aiohttp.ClientSession() as sess:
        async with sess.request(
            method,
            url,
            data=data,
            headers=headers,
            timeout=timeout,
        ) as r:
            r_body = await r.read()

    headers = t.cast(t.Mapping[str, str], r.headers)
    return r.status, Headers(headers), r_body


def _wait_until_api_server_ready(
    host_url: str,
    timeout: float,
    check_interval: float = 1,
    popen: t.Optional["subprocess.Popen[t.Any]"] = None,
) -> bool:
    start_time = time.time()
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    logger.info("Waiting for host %s to be ready..", host_url)
    while time.time() - start_time < timeout:
        try:
            if popen and popen.poll() is not None:
                return False
            elif opener.open(f"http://{host_url}/readyz", timeout=1).status == 200:
                return True
            else:
                time.sleep(check_interval)
        except (
            ConnectionError,
            urllib.error.URLError,
            socket.timeout,
        ) as e:
            logger.info(f"[{e}]retrying to connect to the host {host_url}...")
            logger.error(e)
            time.sleep(check_interval)
    logger.info(
        f"Timed out waiting {timeout} seconds for Server {host_url} to be ready, "
    )
    return False


@cached_contextmanager("{project_path}")
def bentoml_build(
    project_path: str, bentoml_home: str | None = None
) -> t.Generator[Bento, None, None]:
    """
    Build a BentoML project.
    """
    if bentoml_home:
        from bentoml._internal.configuration.containers import BentoMLContainer

        BentoMLContainer.bentoml_home.set(bentoml_home)

    logger.info(f"Building bento: {project_path}")
    from bentoml import bentos

    bento = bentos.build_bentofile(build_ctx=project_path)
    yield bento
    logger.info(f"Deleting bento: {bento.tag}")
    bentos.delete(str(bento.tag))


@cached_contextmanager("{bento_tag}, {image_tag}")
def bentoml_containerize(
    bento_tag: str | Tag, image_tag: str | None = None, bentoml_home: str | None = None
) -> t.Generator[str, None, None]:
    """
    Build the docker image from a saved bento, yield the docker image tag
    """
    if bentoml_home:
        from bentoml._internal.configuration.containers import BentoMLContainer

        BentoMLContainer.bentoml_home.set(bentoml_home)

    bento_tag = Tag.from_taglike(bento_tag)
    if image_tag is None:
        image_tag = bento_tag.name

    from bentoml import bentos

    logger.info(f"Building bento server docker image: {bento_tag}")
    bentos.containerize(str(bento_tag), docker_image_tag=image_tag)

    yield image_tag

    logger.info(f"Removing bento server docker image: {image_tag}")
    subprocess.call(["docker", "rmi", image_tag])


@cached_contextmanager("{image_tag}, {config_file}, {grpc}")
def run_bento_server_in_docker(
    image_tag: str,
    config_file: str | None = None,
    grpc: bool = False,
    timeout: float = 40,
):
    """
    Launch a bentoml service container from a docker image, yield the host URL
    """
    container_name = f"bentoml-test-{image_tag}-{hash(config_file)}"
    with reserve_free_port() as port:
        pass

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--publish",
        f"{port}:3000",
        "--env",
        "BENTOML_LOG_STDOUT=true",
        "--env",
        "BENTOML_LOG_STDERR=true",
    ]

    if config_file is not None:
        cmd.extend(["--env", "BENTOML_CONFIG=/home/bentoml/bentoml_config.yml"])
        cmd.extend(
            ["-v", f"{os.path.abspath(config_file)}:/home/bentoml/bentoml_config.yml"]
        )
    if grpc:
        prom_port = BentoMLContainer.grpc.metrics.port.get()
        cmd.extend(["-p", f"{prom_port}:{prom_port}"])

    cmd.append(image_tag)

    if grpc:
        cmd.extend(["--grpc", "--enable-reflection"])

    logger.info(f"Running API server docker image: {cmd}")
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        encoding="utf-8",
    ) as proc:
        try:
            host_url = f"127.0.0.1:{port}"
            if _wait_until_api_server_ready(host_url, timeout, popen=proc):
                yield host_url
            else:
                raise RuntimeError(
                    f"API server {host_url} failed to start within {timeout} seconds"
                )
        finally:
            subprocess.call(["docker", "stop", container_name])
    time.sleep(1)


@contextmanager
def run_bento_server(
    bento: str,
    workdir: str | None = None,
    grpc: bool = False,
    config_file: str | None = None,
    dev_server: bool = False,
    timeout: float = 90,
):
    """
    Launch a bentoml service directly by the bentoml CLI, yields the host URL.
    """
    workdir = workdir if workdir is not None else "./"
    my_env = os.environ.copy()
    if config_file is not None:
        my_env["BENTOML_CONFIG"] = os.path.abspath(config_file)
    with reserve_free_port() as port:
        cmd = [sys.executable, "-m", "bentoml", "serve"]
        if not dev_server:
            cmd += ["--production"]
        if port:
            cmd += ["--port", f"{port}"]
        cmd += [bento]
        cmd += ["--working-dir", workdir]

    if grpc:
        cmd += ["--grpc", "--enable-reflection"]

    logger.info(f"Running command: `{cmd}`")
    p = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        env=my_env,
        encoding="utf-8",
    )

    try:
        host_url = f"127.0.0.1:{port}"
        assert _wait_until_api_server_ready(host_url, timeout=timeout, popen=p)
        yield host_url
    finally:
        kill_subprocess_tree(p)
        p.communicate()


def _start_mitm_proxy(port: int) -> None:
    import uvicorn  # type: ignore

    from .utils import http_proxy_app

    logger.info(f"proxy serer listen on {port}")
    uvicorn.run(http_proxy_app, port=port)  # type: ignore (not using ASGI3Application)


@contextmanager
def run_bento_server_distributed(
    bento_tag: str | Tag,
    config_file: str | None = None,
    bentoml_home: str | None = None,
    grpc: bool = False,
    timeout: float = 90,
):
    """
    Launch a bentoml service as a simulated distributed environment(Yatai), yields the host URL.
    """
    with reserve_free_port() as proxy_port:
        pass

    logger.warning(f"Starting proxy on port {proxy_port}")
    proxy_process = multiprocessing.Process(
        target=_start_mitm_proxy,
        args=(proxy_port,),
    )
    proxy_process.start()

    copied = os.environ.copy()

    # to ensure yatai specified headers BP100
    copied["YATAI_BENTO_DEPLOYMENT_NAME"] = "sdfasdf"
    copied["YATAI_BENTO_DEPLOYMENT_NAMESPACE"] = "yatai"
    copied["HTTP_PROXY"] = f"http://127.0.0.1:{proxy_port}"

    if config_file is not None:
        copied["BENTOML_CONFIG"] = os.path.abspath(config_file)

    import yaml

    import bentoml

    if bentoml_home:
        from bentoml._internal.configuration.containers import BentoMLContainer

        BentoMLContainer.bentoml_home.set(bentoml_home)

    bento_service = bentoml.bentos.get(bento_tag)

    path = bento_service.path

    with open(os.path.join(path, "bento.yaml"), "r", encoding="utf-8") as f:
        bentofile = yaml.safe_load(f)

    runner_map = {}
    processes: t.List[subprocess.Popen[str]] = []

    for runner in bentofile["runners"]:
        with reserve_free_port() as port:
            runner_map[runner["name"]] = f"tcp://127.0.0.1:{port}"
            cmd = [
                sys.executable,
                "-m",
                "bentoml",
                "start-runner-server",
                str(bento_tag),
                "--runner-name",
                runner["name"],
                "--host",
                "127.0.0.1",
                "--port",
                f"{port}",
                "--working-dir",
                path,
            ]
            logger.info(f"Running command: `{cmd}`")

        processes.append(
            subprocess.Popen(
                cmd,
                encoding="utf-8",
                stderr=subprocess.STDOUT,
                env=copied,
            )
        )

    with reserve_free_port() as server_port:
        if grpc:
            import json

            bind = f"tcp://127.0.0.1:{server_port}"
            copied["BENTOML_RUNNER_MAP"] = json.dumps(runner_map)
            cmd = [
                sys.executable,
                "-m",
                "bentoml_cli.server.grpc_api_server",
                str(bento_tag),
                "--bind",
                bind,
                "--working-dir",
                path,
                "--enable-reflection",
            ]
        else:
            args_pairs = [
                ("--remote-runner", f"{runner['name']}={runner_map[runner['name']]}")
                for runner in bentofile["runners"]
            ]
            cmd = [
                sys.executable,
                "-m",
                "bentoml",
                "start-http-server",
                str(bento_tag),
                "--host",
                "127.0.0.1",
                "--port",
                f"{server_port}",
                "--working-dir",
                path,
                *itertools.chain.from_iterable(args_pairs),
            ]
        logger.info(f"Running command: `{cmd}`")

    processes.append(
        subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            env=copied,
        )
    )
    try:
        host_url = f"127.0.0.1:{server_port}"
        _wait_until_api_server_ready(host_url, timeout=timeout)
        yield host_url
    finally:
        for p in processes:
            kill_subprocess_tree(p)
        for p in processes:
            p.communicate()
        proxy_process.terminate()
        proxy_process.join()


@cached_contextmanager(
    "{bento_name}, {project_path}, {config_file}, {deployment_mode}, {bentoml_home}, {grpc}"
)
def host_bento(
    bento_name: t.Union[str, Tag, None] = None,
    project_path: str = ".",
    config_file: str | None = None,
    deployment_mode: str = "standalone",
    bentoml_home: str | None = None,
    grpc: bool = False,
    clean_context: contextlib.ExitStack | None = None,
) -> t.Generator[str, None, None]:
    """
    Host a bentoml service, yields the host URL.

    Args:
        bento: a bento tag or `module_path:service`
        project_path: the path to the project directory
        config_file: the path to the config file
        deployment_mode: the deployment mode, one of `standalone`, `docker` or `distributed`
        clean_context: a contextlib.ExitStack to clean up the intermediate files,
                       like docker image and bentos. If None, it will be created. Used for reusing
                       those files in the same test session.
    """
    import bentoml

    if clean_context is None:
        clean_context = contextlib.ExitStack()
        clean_on_exit = True
    else:
        clean_on_exit = False

    try:
        logger.info(
            f"starting bento server {bento_name} at {project_path} "
            f"with config file {config_file} "
            f"in {deployment_mode} mode..."
        )
        if bento_name is None or not bentoml.list(bento_name):
            bento = clean_context.enter_context(
                bentoml_build(project_path, bentoml_home=bentoml_home)
            )
        else:
            bento = bentoml.get(bento_name)

        bento_tag = bento.tag

        if deployment_mode == "docker":
            image_tag = clean_context.enter_context(
                bentoml_containerize(bento_tag, bentoml_home=bentoml_home)
            )
            with run_bento_server_in_docker(image_tag, config_file, grpc=grpc) as host:
                yield host
        elif deployment_mode == "standalone":
            with run_bento_server(
                bento.path,
                config_file=config_file,
                workdir=project_path,
                grpc=grpc,
            ) as host:
                yield host
        elif deployment_mode == "distributed":
            with run_bento_server_distributed(
                bento.path,
                config_file=config_file,
                grpc=grpc,
                bentoml_home=bentoml_home,
            ) as host:
                yield host
        else:
            raise ValueError(f"Unknown deployment mode: {deployment_mode}")
    finally:
        logger.info("shutting down bento server...")
        if clean_on_exit:
            logger.info("Cleaning up...")
            clean_context.close()
