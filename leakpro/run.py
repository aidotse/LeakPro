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
"""Run scripts."""
from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA


def run_gia_attack(attack_object: AbstractGIA, experiment_name: str = "GIA",
                path:str = "./leakpro_output/results", save:bool = True) -> None:
    """Runs InvertingGradients."""
    attack_object.prepare_attack()
    result_gen = attack_object.run_attack()
    for _, _, result_object in result_gen:
        if result_object is not None:
            break
    if save:
        result_object.save(name=experiment_name, path=path, config=attack_object.get_configs())
    return result_object
