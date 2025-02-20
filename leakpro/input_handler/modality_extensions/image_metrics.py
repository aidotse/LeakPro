"""ImageMetrics class for computing image metrics."""
import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.attacks.utils.generator_handler import GeneratorHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.logger import logger


class ImageMetrics:
    """Class for computing image metrics."""

    def __init__(self,  handler: AbstractInputHandler, generator_handler: GeneratorHandler, configs: dict) -> None:
        """Initialize the ImageMetrics class."""
        super().__init__(handler)
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
            "accuracy": self.compute_accuracy(),
        }

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
        tests = self.configs.get("tests", [])
        # If tests empty, return
        if not tests:
            logger.warning("No tests specified in the config.")
            return

        self.generate_samples()
        for test in tests:
            if test in self.test_dict:
                self.test_dict[test]()
            else:
                logger.warning(f"Test {test} not found in the test dictionary.")

    def generate_samples(self) -> None:
        """Generate samples from the generator."""
        logger.info("Generating samples for the audited classes.")
        self.generator.eval()
        self.generated_samples = []
        for i in range(self.num_audited_classes):
            self.generated_samples.append(self.generator_handler.sample_from_generator(self.generator,
                                                                                       self.num_audited_classes,
                                                                                       self.num_class_samples,
                                                                                       self.device,
                                                                                       self.generator.dim_z,
                                                                                       label=i))

    def compute_accuracy(self) -> None:
        """Compute accuracy for generated samples."""
        logger.info("Computing accuracy for generated samples.")
        self.evaluation_model.eval()
        correct_predictions = []
        for i in range(self.num_audited_classes):
            for j in range(self.num_class_samples):
                with torch.no_grad():
                    output = self.evaluation_model(self.generated_samples[i][j])
                    prediction = torch.argmax(output, dim=1)
                    correct_predictions.append(prediction == i)
        self.accuracy = np.mean(correct_predictions)
        self.accuracy_std = np.std(correct_predictions)
        logger.info(f"Accuracy: {self.accuracy.item()}")

        self.results["accuracy"] = self.accuracy.item()
        self.results["accuracy_std"] = self.accuracy_std.item()