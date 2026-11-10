# OpenShell sandbox image for deep-agents-demo.
#
# NVIDIA's default sandbox image has no Python or Git. Deep Agents runs its file
# tools (read, edit, ls, grep, glob) as python3 scripts, and the agent needs Git
# and uv for coding tasks, so they are baked in here: the sandbox itself is not
# root and policy.yaml blocks package mirrors, so it cannot install them later.
#
# Build with the Docker engine your OpenShell gateway uses, then create the sandbox:
#   docker build -t deep-agents-demo-sandbox:0.1 sandbox/
#   openshell sandbox create --name deep-agents-demo --from deep-agents-demo-sandbox:0.1 \
#     --policy policy.yaml --detach
FROM nvcr.io/nvidia/base/ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python-is-python3 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Paths match the binaries that policy.yaml allows to reach PyPI.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/

# No WORKDIR: OpenShell then uses /sandbox, which is also the agent's home.
