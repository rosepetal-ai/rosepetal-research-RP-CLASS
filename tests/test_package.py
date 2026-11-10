# RP-CLASS - Copyright (C) 2026 Rosepetal AI
# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License
# See LICENSE for licensing terms and commercial inquiries.
"""The repository itself: one version written in two places, and the licence header in every source file."""
import re
from pathlib import Path

import rpclass

ROOT = Path(__file__).resolve().parents[1]
HEADER = ['# RP-CLASS - Copyright (C) 2026 Rosepetal AI',
          '# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Rosepetal-Enterprise-License',
          '# See LICENSE for licensing terms and commercial inquiries.']
NOT_SOURCES = {'.git', 'build', 'dist', 'runs', '.venv', 'venv', '__pycache__', '.pytest_cache'}


def test_the_version_is_the_same_in_pyproject_and_the_package():
    """pyproject.toml and rpclass/__init__.py both carry the version; a release bumps them together."""
    version = re.search(r'^version = "([^"]+)"', (ROOT / 'pyproject.toml').read_text(), re.M).group(1)
    assert version == rpclass.__version__


def test_every_python_file_starts_with_the_licence_header():
    """Copyright, SPDX expression and where to read the licence, as the first three lines of every .py file."""
    sources = [p for p in ROOT.rglob('*.py')
               if not NOT_SOURCES & set(p.relative_to(ROOT).parts)
               and not any(part.endswith('.egg-info') for part in p.parts)]
    missing = [str(p.relative_to(ROOT)) for p in sources if p.read_text().splitlines()[:3] != HEADER]
    assert sources and not missing, f'files without the licence header: {missing}'
