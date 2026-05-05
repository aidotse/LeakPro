#
# Copyright 2023-2026 AI Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Test for the image input handler."""

import os
import shutil

import pandas as pd
import pytest
import yaml
from dotmap import DotMap

from leakpro import LeakPro
from leakpro.synthetic_data_attacks.anonymeter.evaluators import singling_out_evaluator as singl_ev
from leakpro.tests.constants import STORAGE_PATH, get_audit_config
from leakpro.tests.input_handler.image_input_handler import ImageInputHandler
from leakpro.tests.input_handler.image_utils import setup_image_test
from leakpro.tests.input_handler.tabular_input_handler import TabularInputHandler
from leakpro.tests.input_handler.tabular_utils import setup_tabular_test


@pytest.fixture(autouse=True)
def patch_singling_out_evaluator_convert():
    """Patch convert_df_numerical_columns_to_categories_with_threshold for test compatibility.

    This fixture ensures the function modifies DataFrames in-place (as expected by tests)
    rather than returning a copy. It also skips boolean columns to avoid dtype errors.
    """
    original_func = singl_ev.convert_df_numerical_columns_to_categories_with_threshold

    def patched_func(*, df: pd.DataFrame, threshold: int) -> pd.DataFrame:
        for col in df.columns:
            if (
                pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_bool_dtype(df[col])
                and (df[col].nunique() <= threshold)
            ):
                df[col] = df[col].astype("category")
        return df

    singl_ev.convert_df_numerical_columns_to_categories_with_threshold = patched_func
    yield
    singl_ev.convert_df_numerical_columns_to_categories_with_threshold = original_func


@pytest.fixture(scope="session")
def manage_storage_directory():
    """Fixture to create and remove the storage directory."""

    # Setup: Create the folder at the start of the test session
    os.makedirs(STORAGE_PATH, exist_ok=True)

    # Yield control back to the test session
    yield

    # Teardown: Remove the folder and its contents at the end of the session
    if os.path.exists(STORAGE_PATH):
        shutil.rmtree(STORAGE_PATH)


@pytest.fixture
def image_handler(manage_storage_directory) -> ImageInputHandler:
    """Fixture for the image input handler to be shared between many tests."""

    config = DotMap()
    config.target = setup_image_test()
    config.target.dpsgd_path = f"{STORAGE_PATH}/dummy_dpsgd_model.pt"
    config.audit = get_audit_config()
    config.audit.data_modality = "image"
    # save config to file
    config_path = f"{STORAGE_PATH}/image_test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.toDict(), f)

    leakpro = LeakPro(ImageInputHandler, config_path)
    handler = leakpro.handler
    handler.configs = DotMap(handler.configs)

    # Yield control back to the test session
    return handler


@pytest.fixture
def tabular_handler(manage_storage_directory) -> TabularInputHandler:
    """Fixture for the image input handler to be shared between many tests."""

    config = DotMap()
    config.target = setup_tabular_test()
    config.target.dpsgd_path = f"{STORAGE_PATH}/dummy_dpsgd_model.pt"
    config.audit = get_audit_config()
    config.audit.data_modality = "tabular"
    # save config to file
    config_path = f"{STORAGE_PATH}/tabular_test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.toDict(), f)

    leakpro = LeakPro(TabularInputHandler, config_path)
    handler = leakpro.handler
    handler.configs = DotMap(handler.configs)

    # Yield control back to the test session
    return handler
