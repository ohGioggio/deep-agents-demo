---
name: sandbox
description: How to work inside the OpenShell sandbox (working directory /sandbox). Read this before running commands, installing Python packages, cloning repositories, or when a command fails with "Permission denied" or HTTP 403.
---
# Working in the OpenShell sandbox

Your `execute` tool and file tools run in an isolated Linux sandbox, not on the
user's computer. A policy enforced by the kernel and a network proxy decides
what you can reach. You cannot change that policy; only the user can.

## The environment

- Ubuntu 24.04. You are user `ubuntu`, not root. There is no `sudo` and no `apt`.
- Working and home directory: `/sandbox`. Keep code, data and outputs there.
  `/tmp` is writable for scratch files. Everything else is read-only.
- Installed: `python` / `python3` (3.12), `git`, `uv`. Not installed: `pip`,
  `curl`, `wget`. Use `uv` for every Python package task.
- No display. Write images and reports to files.
- Each command times out after 120 seconds. Split long jobs into steps.
- Files persist until the user recreates the sandbox. Copy nothing important to
  `/tmp`.

## Network: what is allowed

| Allowed | Blocked |
| --- | --- |
| Installing packages from PyPI with `uv` | Uploading to PyPI |
| `git clone` / `git fetch` over HTTPS from github.com | `git push`, the GitHub API, other Git hosts |
| | Every other website, API and port |

Only `uv` and Python may reach PyPI, and only Git may reach GitHub.

## Installing Python packages

`pip` is not installed and `pip install` will fail. Use `uv` instead:

- **Reusable environment for a task** (preferred):
  ```bash
  cd /sandbox && uv venv            # once; creates /sandbox/.venv
  uv pip install matplotlib numpy   # installs into /sandbox/.venv
  .venv/bin/python script.py
  ```
- **One-off script:** `uv run --with requests python script.py`
- **A project with dependencies:** `uv init myproj && cd myproj && uv add rich`,
  then `uv run python main.py`.
- **A cloned repository:** if it has `pyproject.toml`, run `uv sync`, then
  `uv run pytest` (or the project's own test command). If it only has
  `requirements.txt`, run `uv venv && uv pip install -r requirements.txt`.

Use the system Python 3.12. `uv python install` and Python downloads from
GitHub releases are blocked, so do not request another Python version.

## Plots and images

Install matplotlib with uv and use the non-interactive backend:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
...
plt.savefig("/sandbox/plot.png", dpi=150)
```

Tell the user the output path. Do not fall back to hand-written SVG just because
`pip` is missing: install the library with uv.

## When something is blocked

A blocked connection fails fast with `Permission denied (os error 13)`,
`Errno 13`, "Couldn't connect to server", or an HTTP `403 Forbidden` from the
proxy. When you see one:

1. Do not retry the same request, and do not try another tool, host, port or
   mirror to get around it. Working around the policy is not allowed.
2. If a PyPI package or a GitHub repository can do the job, use that instead.
3. Otherwise stop and tell the user what was blocked, for example
   "the sandbox policy blocks api.example.com:443". They can allow it by adding
   a rule to `policy.yaml` and running
   `openshell policy set deep-agents-demo --policy policy.yaml --wait`.

## Good habits

- Check before assuming: `command -v <tool>` and `ls`.
- Run tests after editing code and report the command and its result.
- Only claim a file exists after a command or tool has created it successfully.
