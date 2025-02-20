"""ImageMetrics class for computing image metrics."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from leakpro.attacks.utils.generator_handler import GeneratorHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.logger import logger


class ImageMetrics:
    """Class for computing image metrics."""

    def __init__(self,  handler: AbstractInputHandler, generator_handler: GeneratorHandler, configs: dict) -> None:
        """Initialize the ImageMetrics class."""
        self.handler = handler
        self.generator_handler = generator_handler
        self.generator = self.generator_handler.get_generator()
        self.evaluation_model = self.handler.target_model # TODO: Change to evaluation model from configs
        self.target_model = self.handler.target_model
        logger.info(f"Target model: {self.target_model}")
        logger.info(f"Evaluation model: {self.evaluation_model}")
        logger.info("Configuring ImageMetrics")
        self._configure_metrics(configs)
        # Compute desired metrics from configs
        self.test_dict = {
            "accuracy": self.compute_accuracy,
        }
        logger.info(configs)
        self.results = {}
        self.metric_scheduler()

    def _configure_metrics(self, configs: dict) -> None:
        """Configure the metrics parameters.

        Args:
        ----
            configs (dict): Configuration parameters for the metrics.

        """
        self.configs = configs
        self.num_classes = configs.get("num_classes")
        self.batch_size = configs.get("batch_size", 32)
        self.num_class_samples = configs.get("num_class_samples", 1)
        self.num_audited_classes = configs.get("num_audited_classes", self.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def metric_scheduler(self) -> None:
        """Schedule the metrics to be computed."""
        tests = self.configs.get("metrics", [])
        # If tests empty, return
        if not tests:
            logger.warning("No tests specified in the config.")
            return

        for test in tests:
            if test in self.test_dict:
                self.test_dict[test]()
            else:
                logger.warning(f"Test {test} not found in the test dictionary.")

    def compute_accuracy(self) -> None:
        """Compute accuracy for generated samples."""
        logger.info("Computing accuracy for generated samples.")
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)
        self.generator.eval()
        self.generator.to(self.device)
        correct_predictions = []
        for i in range(self.num_audited_classes):
            generated_sample = self.generator_handler.sample_from_generator(self.generator,
                                                                            self.num_audited_classes,
                                                                            self.num_class_samples,
                                                                            self.device,
                                                                            self.generator.dim_z,
                                                                            label=i)
            for j in range(self.num_class_samples):
                with torch.no_grad():
                    output = self.evaluation_model(generated_sample[j])
                    prediction = torch.argmax(output, dim=1)
                    correct_predictions.append(prediction == i)
        self.accuracy = torch.mean(torch.cat(correct_predictions).float())
        self.accuracy_std = torch.std(torch.cat(correct_predictions).float())
        logger.info(f"Accuracy: {self.accuracy.item()}")

        self.results["accuracy"] = self.accuracy.item()
        self.results["accuracy_std"] = self.accuracy_std.item()
