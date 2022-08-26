# pylint: disable=redefined-outer-name,not-context-manager
from __future__ import annotations

import os
import sys
import time
import socket
import typing as t
import urllib
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


def http_server_warmup(
    host_url: str,
    timeout: float,
    check_interval: float = 1,
    popen: t.Optional["subprocess.Popen[t.Any]"] = None,
) -> bool:
    start_time = time.time()
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    print("Waiting for host %s to be ready.." % host_url)
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
            print(f"[{e}] Retrying to connect to the host {host_url}...")
            time.sleep(check_interval)
    print(f"Timed out waiting {timeout} seconds for Server {host_url} to be ready.")
    return False


@cached_contextmanager("{project_path}")
def bentoml_build(project_path: str) -> t.Generator[Bento, None, None]:
    """
    Build a BentoML project.
    """
    from bentoml import bentos

    bento = bentos.build_bentofile(build_ctx=project_path)
    yield bento


@cached_contextmanager("{bento_tag}, {image_tag}")
def bentoml_containerize(
    bento_tag: str | Tag, image_tag: str | None = None
) -> t.Generator[str, None, None]:
    """
    Build the docker image from a saved bento, yield the docker image tag
    """
    from bentoml import bentos

    bento_tag = Tag.from_taglike(bento_tag)
    if image_tag is None:
        image_tag = bento_tag.name
    print(f"Building bento server docker image: {bento_tag}")
    bentos.containerize(str(bento_tag), docker_image_tag=image_tag, progress="plain")
    yield image_tag
    print(f"Removing bento server docker image: {image_tag}")
    subprocess.call(["docker", "rmi", image_tag])


@cached_contextmanager("{image_tag}, {config_file}, {grpc}")
def run_bento_server_in_docker(
    image_tag: str,
    config_file: str | None = None,
    grpc: bool = False,
    timeout: float = 40,
    host: str = "0.0.0.0",
):
    """
    Launch a bentoml service container from a docker image, yield the host URL
    """
    container_name = f"bentoml-test-{image_tag}-{hash(config_file)}"
    with reserve_free_port(enable_so_reuseport=grpc) as port:
        pass

    bind_port = "3000" if not grpc else "50051"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--publish",
        f"{port}:{bind_port}",
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
        from bentoml._internal.configuration.containers import BentoMLContainer

        with reserve_free_port(enable_so_reuseport=True) as prometheus_port:
            pass

        prom_port = BentoMLContainer.grpc.metrics.port.get()
        cmd.extend(["-p", f"{prometheus_port}:{prom_port}"])
        cmd.extend(["--env", "BENTOML_USE_GRPC=true"])
    cmd.append(image_tag)
    if grpc:
        cmd.extend(["--enable-reflection"])
    print(f"Running API server docker image: '{' '.join(cmd)}'")
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        encoding="utf-8",
    ) as proc:
        try:
            host_url = f"{host}:{port}"
            if grpc:
                yield host_url
            elif http_server_warmup(host_url, timeout, popen=proc):
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
    grpc: bool = False,
    config_file: str | None = None,
    timeout: float = 90,
    host: str = "0.0.0.0",
):
    """
    Launch a bentoml service directly by the bentoml CLI, yields the host URL.
    """
    my_env = os.environ.copy()
    if config_file is not None:
        my_env["BENTOML_CONFIG"] = os.path.abspath(config_file)

    with reserve_free_port(host=host, enable_so_reuseport=grpc) as server_port:
        cmd = [
            sys.executable,
            "-m",
            "bentoml",
            "serve-grpc" if grpc else "serve",
            "--production",
            "--port",
            f"{server_port}",
        ]
        if grpc:
            cmd += ["--host", f"{host}", "--enable-reflection"]
    cmd += [bento]

    print(f"Running command: '{' '.join(cmd)}'")
    p = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        env=my_env,
        encoding="utf-8",
    )
    try:
        host_url = f"{host}:{server_port}"
        if not grpc:
            assert http_server_warmup(host_url, timeout=timeout, popen=p)
        yield host_url
    finally:
        kill_subprocess_tree(p)
        p.communicate()


def _start_mitm_proxy(port: int) -> None:
    import uvicorn  # type: ignore

    from .utils import http_proxy_app

    print(f"Proxy server listen on {port}")
    uvicorn.run(http_proxy_app, port=port)  # type: ignore (not using ASGI3Application)


@contextmanager
def run_bento_server_distributed(
    bento_tag: str | Tag,
    config_file: str | None = None,
    grpc: bool = False,
    timeout: float = 90,
    host: str = "0.0.0.0",
):
    """
    Launch a bentoml service as a simulated distributed environment(Yatai), yields the host URL.
    """
    with reserve_free_port() as proxy_port:
        pass

    print(f"Starting proxy on port {proxy_port}")
    proxy_process = multiprocessing.Process(
        target=_start_mitm_proxy,
        args=(proxy_port,),
    )
    proxy_process.start()

    copied = os.environ.copy()

    # to ensure yatai specified headers BP100
    copied["YATAI_BENTO_DEPLOYMENT_NAME"] = "test-deployment"
    copied["YATAI_BENTO_DEPLOYMENT_NAMESPACE"] = "yatai"
    copied["HTTP_PROXY"] = f"http://{host}:{proxy_port}"

    if config_file is not None:
        copied["BENTOML_CONFIG"] = os.path.abspath(config_file)

    import yaml

    import bentoml

    bento_service = bentoml.bentos.get(bento_tag)

    path = bento_service.path

    with open(os.path.join(path, "bento.yaml"), "r", encoding="utf-8") as f:
        bentofile = yaml.safe_load(f)

    runner_map = {}
    processes: t.List[subprocess.Popen[str]] = []

    for runner in bentofile["runners"]:
        with reserve_free_port() as port:
            runner_map[runner["name"]] = f"tcp://{host}:{port}"
            cmd = [
                sys.executable,
                "-m",
                "bentoml",
                "start-runner-server",
                str(bento_tag),
                "--runner-name",
                runner["name"],
                "--host",
                host,
                "--port",
                f"{port}",
                "--working-dir",
                path,
            ]
            print(f"Running command: '{' '.join(cmd)}'")

        processes.append(
            subprocess.Popen(
                cmd,
                encoding="utf-8",
                stderr=subprocess.STDOUT,
                env=copied,
            )
        )
    runner_args = [
        ("--remote-runner", f"{runner['name']}={runner_map[runner['name']]}")
        for runner in bentofile["runners"]
    ]
    cmd = [
        sys.executable,
        "-m",
        "bentoml",
        "start-http-server" if not grpc else "start-grpc-server",
        str(bento_tag),
        "--host",
        host,
        "--working-dir",
        path,
        *itertools.chain.from_iterable(runner_args),
    ]
    with reserve_free_port(host=host, enable_so_reuseport=grpc) as server_port:
        cmd.extend(["--port", f"{server_port}"])
        if grpc:
            cmd.append("--enable-reflection")
    print(f"Running command: '{' '.join(cmd)}'")

    processes.append(
        subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            env=copied,
        )
    )
    try:
        host_url = f"{host}:{server_port}"
        if not grpc:
            http_server_warmup(host_url, timeout=timeout)
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
    bento_name: str | Tag | None = None,
    project_path: str = ".",
    config_file: str | None = None,
    deployment_mode: str = "standalone",
    bentoml_home: str | None = None,
    grpc: bool = False,
    clean_context: contextlib.ExitStack | None = None,
    host: str = "0.0.0.0",
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
    if bentoml_home:
        from bentoml._internal.configuration.containers import BentoMLContainer

        BentoMLContainer.bentoml_home.set(bentoml_home)

    try:
        print(
            f"Starting bento server {bento_name} at '{project_path}' {'with config file '+config_file if config_file else ''} in {deployment_mode} mode..."
        )
        if bento_name is None or not bentoml.list(bento_name):
            bento = clean_context.enter_context(bentoml_build(project_path))
        else:
            bento = bentoml.get(bento_name)

        if deployment_mode == "standalone":
            with run_bento_server(
                bento.path,
                config_file=config_file,
                grpc=grpc,
                host=host,
            ) as host:
                yield host
        elif deployment_mode == "docker":
            image_tag = clean_context.enter_context(bentoml_containerize(bento.tag))
            with run_bento_server_in_docker(
                image_tag,
                config_file,
                grpc=grpc,
                host=host,
            ) as host:
                yield host
        elif deployment_mode == "distributed":
            with run_bento_server_distributed(
                bento.path,
                config_file=config_file,
                grpc=grpc,
                host=host,
            ) as host:
                yield host
        else:
            raise ValueError(f"Unknown deployment mode: {deployment_mode}")
    finally:
        print("Shutting down bento server...")
        if clean_on_exit:
            print("Cleaning up...")
            clean_context.close()
