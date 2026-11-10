#!/usr/bin/env bash
# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
#
# One-time setup so this clone runs the secret scan on every commit (git never installs hooks on clone).
# Safe to run again, for example after the gitleaks rev in .pre-commit-config.yaml is bumped.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v pre-commit >/dev/null 2>&1; then
  echo "pre-commit not found; installing it"
  if command -v pipx >/dev/null 2>&1; then
    pipx install pre-commit
  elif command -v pip3 >/dev/null 2>&1; then
    pip3 install --user pre-commit
  else
    echo "error: pipx or pip3 is needed to install pre-commit (https://pre-commit.com/#install)" >&2
    exit 1
  fi
fi

pre-commit install
echo "done: gitleaks now runs on every 'git commit' (scan the whole tree with: pre-commit run --all-files)"
