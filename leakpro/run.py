"""Run scripts."""
from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA



def run_gia_attack(attack_object: AbstractGIA, experiment_name: str = "GIA",
                path:str = "./leakpro_output/results", save:bool = True) -> None:
    """Runs InvertingGradients."""
    result_gen = attack_object.run_attack()
    for _, _, result_object in result_gen:
        if result_object is not None:
            break
    if save:
        result_object.save(name=experiment_name, path=path, config=attack_object.get_configs())
    if save:
        result_object.save(name=experiment_name, path=path, config=attack_object.get_configs())
    return result_object