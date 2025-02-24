"""ImageMetrics class for computing image metrics."""
import numpy as np
import torch
from scipy.linalg import sqrtm
from torchvision import models, transforms

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
        logger.info("Configuring ImageMetrics")
        self._configure_metrics(configs)
        self.test_dict = {
            "accuracy": self.compute_accuracy,
            "fid" : self.compute_fid,
        }
        logger.info(configs)
        self.results = {}
        # TODO: This loading functionality should not be in generator_handler
        self.private_dataloader = self.generator_handler.get_private_data(self.batch_size)
        # Compute desired metrics from configs
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
            generated_sample = self.generator_handler.sample_from_generator(label=i)
            for j in range(self.num_class_samples):
                with torch.no_grad():
                    output = self.evaluation_model(generated_sample[j])
                    prediction = torch.argmax(output, dim=1)
                    correct_predictions.append(prediction == i)
        correct_predictions = torch.cat(correct_predictions).float()
        self.accuracy = correct_predictions.mean()
        self.accuracy_std = correct_predictions.std() / torch.sqrt(torch.tensor(len(correct_predictions), dtype=torch.float))
        logger.info(f"Accuracy: {self.accuracy.item()}")

        self.results["accuracy"] = self.accuracy.item()
        self.results["accuracy_std"] = self.accuracy_std.item()


    def compute_fid(self) -> None:
        """Compute the Frechet Inception Distance (FID) between real and generated images."""
        logger.info("Computing FID between real and generated samples.")

        # Load InceptionV3 model for feature extraction
        inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        inception_model.fc = torch.nn.Identity()  # Remove final classification layer
        inception_model.eval()
        inception_model.to(self.device)

        # Image transformation for InceptionV3 input
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        def tensor_to_pil(tensor):  # noqa: ANN001, ANN202
            """Convert tensor image (C, H, W) -> PIL Image."""
            tensor = tensor.detach().cpu().clamp(0, 1)  # Ensure values are in [0, 1]
            return transforms.ToPILImage()(tensor)

        def get_features(dataloader, model):  # noqa: ANN001, ANN202
            """Extract features from images using InceptionV3."""
            features = []
            with torch.no_grad():
                for images, _ in dataloader:
                    images = images.to(self.device)
                    pil_images = [tensor_to_pil(img) for img in images]  # Convert each tensor to PIL
                    transformed_images = torch.stack([transform(img) for img in pil_images]).to(self.device)

                    feats = model(transformed_images)
                    features.append(feats)
            return torch.cat(features, dim=0).cpu().numpy()

        def get_generated_features():  # noqa: ANN202
            """Generate fake images and extract features."""
            self.generator.eval()
            features = []
            with torch.no_grad():
                for _ in range(len(self.private_dataloader)):  # Match number of batches
                    fake_images = self.generator_handler.sample_from_generator()[0]

                    pil_images = [tensor_to_pil(img) for img in fake_images]
                    transformed_images = torch.stack([transform(img) for img in pil_images]).to(self.device)

                    feats = inception_model(transformed_images)
                    features.append(feats)
            return torch.cat(features, dim=0).cpu().numpy()

        # Extract features from real and generated images
        real_features = get_features(self.private_dataloader, inception_model)
        fake_features = get_generated_features()

        # Compute mean and covariance of features
        mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

        # Calculate FID score
        diff = mu_real - mu_fake
        covmean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)

        # Store results
        self.results["fid"] = fid_score.item()
        logger.info(f"FID: {fid_score}")

    def compute_knn_dist(self) -> None:
        """Compute the k-NN distance."""
    pass

