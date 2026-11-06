#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The generative priors' optional dependencies are imported only when the feature that needs them runs."""
from __future__ import annotations

import inspect
import sys

import pytest


def test_missing_cma_raises_an_actionable_import_error(monkeypatch):
    from leakpro.attacks.gia_attacks.modular.components.drivers.cma_es import CMAESDriver

    monkeypatch.setitem(sys.modules, "cma", None)   # makes `import cma` raise ImportError
    assert "import cma" in inspect.getsource(CMAESDriver.run)   # guard: the lazy import lives in run()
    with pytest.raises(ImportError, match="pip install cma"):
        CMAESDriver().run(None, None)   # ImportError is raised before any argument is touched
