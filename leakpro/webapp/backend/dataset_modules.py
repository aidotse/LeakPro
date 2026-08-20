#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Make a job's uploaded dataset classes importable before unpickling its data.

A pickled dataset records the module path its class came from, so unpickling
fails with ``No module named ...`` unless that name resolves. The webapp stores
the user's file as ``<job>/dataset_handler.py``, which is not the name it was
written under, so the module has to be registered under the original name too.

One implementation, used by every path that loads a job's dataset: the audit
worker, the PET campaign runner, and the sample-data/sample-image endpoints.
Three copies of this previously drifted apart, and the endpoints' copy existed
only because a bug report showed images failing for models loaded from an
earlier session.
"""

import sys
from pathlib import Path

from leakpro.input_handler.user_imports import import_module_from_file
from leakpro.utils.logger import logger


def register_job_dataset_modules(job_dir: str | Path) -> None:
    """Register ``<job_dir>/dataset_handler.py`` under the names pickles expect.

    Also puts the job directory on ``sys.path``, since a dataset may have been
    pickled from another module sitting next to it. Safe to call repeatedly and
    a no-op when the job has no uploaded handler.
    """
    job_dir = Path(job_dir)
    if str(job_dir) not in sys.path:
        sys.path.insert(0, str(job_dir))

    path = job_dir / "dataset_handler.py"
    if not path.exists():
        return

    try:
        module = import_module_from_file(str(path))
    except Exception as exc:  # noqa: BLE001 - a broken upload must not take the caller down
        logger.warning(f"Could not import {path}: {exc}")
        return

    sys.modules.setdefault("dataset_handler", module)

    # The handler class name encodes the original module name:
    #   CelebADataHandler -> celebA_data_handler
    #   CifarDataHandler  -> cifar_data_handler
    for attr in dir(module):
        obj = getattr(module, attr, None)
        if isinstance(obj, type) and hasattr(obj, "UserDataset") and attr != "AbstractInputHandler":
            original = attr.replace("DataHandler", "_data_handler").replace("InputHandler", "_input_handler")
            original = original[0].lower() + original[1:]
            if original not in sys.modules:
                sys.modules[original] = module
                logger.info(f"Registered {path.name} as '{original}' for unpickling.")
