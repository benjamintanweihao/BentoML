#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
STUBS_GENERATOR="bentoml/stubs-generator"

cd "$GIT_ROOT" || exit 1

main() {
	# Use inline heredoc for even faster build
	# Keeping image as cache should be fine since we only want to generate the stubs.
	if [[ $(docker images --filter=reference="$STUBS_GENERATOR" -q) == "" ]] || test "$(git diff --name-only --diff-filter=d -- "$0")"; then
		docker buildx build --platform=linux/amd64 -t "$STUBS_GENERATOR" --load -f- . <<EOF
# syntax=docker/dockerfile:1.4-labs

FROM --platform=linux/amd64 python:3-slim-bullseye

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workspace

COPY <<-EOT /workspace/requirements.txt
    # Restrict maximum version due to breaking protobuf 4.21.0 changes
    # (see https://github.com/protocolbuffers/protobuf/issues/10051)
    protobuf>=3.5.0, <3.20
    # There is memory leak in later Python GRPC (1.43.0 to be specific),
    # use known working version until the memory leak is resolved in the future
    # (see https://github.com/grpc/grpc/issues/28513)
    grpcio-tools<1.43
    mypy-protobuf
EOT

RUN --mount=type=cache,target=/var/lib/apt \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y -q -y --no-install-recommends --allow-remove-essential bash build-essential ca-certificates

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

EOF
	fi

	echo "Generating gRPC stubs..."
	docker run --rm -it -v "$GIT_ROOT":/workspace --platform=linux/amd64 "$STUBS_GENERATOR" python -m grpc_tools.protoc -I. --grpc_python_out=. --python_out=. --mypy_out=. --mypy_grpc_out=. bentoml/grpc/v1/service.proto bentoml/grpc/v1/service_test.proto
}

if ! [ "${#@}" -eq 0 ]; then
	echo "$0 takes zero arguments"
	exit 1
fi

main
