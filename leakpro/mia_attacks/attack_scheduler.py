from leakpro.mia_attacks.attack_factory import AttackFactory
from leakpro.mia_attacks.attack_objects import AttackObjects
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AttackAbstract


class AttackScheduler:
    def __init__(
        self,
        population,
        train_test_dataset,
        target_model,
        target_model_metadata,
        configs,
        logs_dirname,
        logger,
    ):
        self.attack_list = configs["audit"]["attack_list"]
        self.attacks = []

        attack_objects = AttackObjects(
            population, train_test_dataset, target_model, configs
        )
        attack_utils = AttackUtils(attack_objects)

        for attack_name in self.attack_list:
            try:
                attack = AttackFactory.create_attack(attack_name, attack_utils, configs)
                self.add_attack(attack)
            except ValueError as e:
                print(e)

        self.logs_dirname = logs_dirname
        self.logger = logger

    def add_attack(self, attack: AttackAbstract):
        self.attacks.append(attack)

    def run_attacks(self):
        results = {}
        for attack, attack_type in zip(self.attacks, self.attack_list):
            self.logger.info(f"Preparing attack: {attack_type}")
            attack.prepare_attack()

            self.logger.info(f"Running attack: {attack_type}")

            result = attack.run_attack()
            results[attack_type] = {"attack_object": attack, "result_object": result}

            self.logger.info(f"Finished attack: {attack_type}")
        return results

    def identify_attacks(self, model, dataset):
        # Identify relevant attacks based on adversary setting
        pass
