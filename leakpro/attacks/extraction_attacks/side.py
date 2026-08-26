#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""SIDE surrogate-conditional extraction for unconditional diffusion models."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from leakpro.attacks.extraction_attacks.abstract_extraction import AbstractExtraction, AttackState
from leakpro.attacks.extraction_attacks.classifier import TimeConditionedResNet
from leakpro.attacks.extraction_attacks.configs import SIDEConfig
from leakpro.attacks.extraction_attacks.metrics import l2_band_scores, nearest_reference, pairwise_band_scores
from leakpro.attacks.extraction_attacks.protocols import (
    DiffusionAdapter,
    FeatureTransform,
    PairwiseScore,
    identity_feature_transform,
)
from leakpro.attacks.extraction_attacks.utils import (
    batch_ranges,
    decode_uint8,
    encode_uint8,
    require_authorized,
    resolve_device,
    seeded_torch_rng,
    stable_hash,
    to_zero_one,
    validate_image_batch,
)
from leakpro.reporting.extraction_result import CandidateRecord, ExtractionResult

ClassifierFactory = Callable[[int, int], nn.Module]


class AttackSIDEExtraction(AbstractExtraction):
    """White-box SIDE attack using synthetic clusters and classifier guidance.

    This is Algorithm 1's small-DPM branch: generate a synthetic set, cluster
    frozen features, filter clusters by cohesion, train a
    time-dependent classifier on forward-noised images, and add its log-class
    gradient to the target model score during reverse diffusion.
    """

    def __init__(
        self,
        adapter: DiffusionAdapter,
        feature_extractor: nn.Module,
        configs: SIDEConfig | dict[str, Any],
        *,
        audit_fingerprint: str,
        reference_images: Tensor | None = None,
        feature_transform: FeatureTransform | None = None,
        classifier: nn.Module | None = None,
        classifier_factory: ClassifierFactory | None = None,
        reference_score_fn: PairwiseScore | None = None,
    ) -> None:
        self.adapter = adapter
        self.feature_extractor = feature_extractor
        self.config = configs if isinstance(configs, SIDEConfig) else SIDEConfig(**configs)
        self.configs = self.config
        self.optuna_params = 0
        self.audit_fingerprint = audit_fingerprint
        self.reference_images = reference_images
        self.feature_transform = feature_transform or identity_feature_transform
        self.classifier = classifier
        self.classifier_factory = classifier_factory
        self.reference_score_fn = reference_score_fn
        self.state = AttackState.CREATED
        identity_config = self.config.model_dump(mode="json", exclude={"overwrite_results"})
        result_hash = stable_hash(
            {"audit_fingerprint": self.audit_fingerprint, "config": identity_config},
            length=16,
        )
        self.result_id = f"side-extraction-{result_hash}"
        self.attack_id = self.result_id
        self.device = resolve_device(self.config.compute_device)
        self.synthetic_images_uint8: Tensor | None = None
        self.synthetic_labels: Tensor | None = None
        self.cluster_centroids: Tensor | None = None
        self.cluster_cohesion: list[float] = []
        self.training_history: list[float] = []
        self._references_zero_one: Tensor | None = None
        self._references_metric: Tensor | None = None
        self._sampling_calls = 0
        self._guidance_calls = 0
        self._initialize_trace()

    def description(self) -> dict[str, str]:
        """Return SIDE's paper reference, threat model, and fidelity boundary."""
        return {
            "title": "SIDE: Surrogate Conditional Data Extraction",
            "reference": (
                "Yunhao Chen, Shujie Wang, Difan Zou, Xingjun Ma, SIDE: Surrogate Conditional Data Extraction "
                "from Diffusion Models, AAAI 2026, https://doi.org/10.1609/aaai.v40i1.36972"
            ),
            "summary": "Discover synthetic feature clusters and turn them into time-dependent classifier guidance.",
            "threat_model": "White-box access to target diffusion parameters and forward/reverse diffusion operations.",
            "scope": (
                "Implements Algorithm 1's time-dependent classifier branch for small DPMs. Classifier epochs and the "
                "concrete timestep-conditioned ResNet are recorded implementation choices because the paper does not "
                "fully specify them. The separate Stable-Diffusion LoRA branch is excluded because its conditioning "
                "and code details are insufficient for a model-agnostic faithful implementation."
            ),
        }

    def prepare_attack(self) -> None:
        """Prepare this attack exactly once."""
        self._prepare_once(self._prepare_attack)

    def _prepare_attack(self) -> None:
        """Create surrogate labels and train the time-dependent classifier."""
        require_authorized(self.config.authorized_audit)
        self._validate_preparation_inputs()
        self._prepare_references()

        self._build_synthetic_dataset()
        self._record_trace(
            "synthetic_dataset_complete",
            sampling_calls=self._sampling_calls,
            synthetic_samples=self.config.synthetic_samples,
        )
        self._fit_surrogate_clusters()
        self._record_trace(
            "clustering_complete",
            requested_clusters=self.config.clusters,
            retained_clusters=len(self.cluster_cohesion),
        )
        with seeded_torch_rng(self.device, self.config.random_seed):
            self._build_classifier()
            self._train_classifier()
        self._record_trace(
            "classifier_training_complete",
            epochs=self.config.classifier_epochs,
            final_loss=self.training_history[-1],
        )
        self._record_trace("prepared")

    def _validate_preparation_inputs(self) -> None:
        """Reject malformed white-box components before target sampling."""
        if not isinstance(self.adapter, DiffusionAdapter):
            raise TypeError("adapter does not satisfy the DiffusionAdapter protocol.")
        self.adapter.validate_side_capabilities()
        if (
            not isinstance(self.adapter.num_timesteps, int)
            or isinstance(self.adapter.num_timesteps, bool)
            or self.adapter.num_timesteps < 2
        ):
            raise ValueError("adapter.num_timesteps must be at least 2.")
        if (
            not isinstance(self.adapter.image_shape, tuple)
            or len(self.adapter.image_shape) != 3
            or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in self.adapter.image_shape)
        ):
            raise ValueError("adapter.image_shape must be a positive CHW tuple.")
        if not isinstance(self.feature_extractor, nn.Module):
            raise TypeError("feature_extractor must be a torch.nn.Module.")
        self._feature_extractor_dtype()
        if not callable(self.feature_transform):
            raise TypeError("feature_transform must be callable.")
        if self.classifier is not None and not isinstance(self.classifier, nn.Module):
            raise TypeError("classifier must be a torch.nn.Module.")
        if self.classifier is not None:
            self._classifier_dtype()
        if self.classifier_factory is not None and not callable(self.classifier_factory):
            raise TypeError("classifier_factory must be callable.")
        if self.reference_score_fn is not None and not callable(self.reference_score_fn):
            raise TypeError("reference_score_fn must be callable.")

    def _prepare_references(self) -> None:
        """Validate reference-dependent metrics before target sampling."""
        if self.reference_images is not None:
            references = validate_image_batch(self.reference_images, self.adapter.image_shape)
            self._references_metric = references.detach().cpu()
            self._references_zero_one = to_zero_one(references.detach().cpu(), self.config.image_range)
        if self.config.similarity_bands and self.reference_score_fn is None:
            raise ValueError("similarity_bands requires a reference_score_fn.")
        if (self.config.l2_bands or self.config.similarity_bands) and self.reference_images is None:
            raise ValueError("Reference score bands require reference_images.")

    def _build_synthetic_dataset(self) -> None:
        stored_images: list[Tensor] = []
        stored_features: list[Tensor] = []
        self.feature_extractor.to(self.device).eval()
        feature_dtype = self._feature_extractor_dtype()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad_(False)
        for batch_index, (start, end) in enumerate(
            batch_ranges(self.config.synthetic_samples, self.config.synthetic_batch_size)
        ):
            batch = self.adapter.sample(
                end - start,
                conditions=None,
                seed=self.config.random_seed + batch_index,
            )
            self._sampling_calls += 1
            batch = validate_image_batch(batch, self.adapter.image_shape, expected_count=end - start)
            zero_one = to_zero_one(batch.detach(), self.config.image_range)
            stored_images.append(encode_uint8(zero_one))
            with torch.no_grad():
                transformed = self.feature_transform(zero_one.to(self.device))
                if not isinstance(transformed, Tensor):
                    raise TypeError("feature_transform must return a torch.Tensor.")
                features = self.feature_extractor(transformed.to(device=self.device, dtype=feature_dtype))
            if not isinstance(features, Tensor):
                raise TypeError("feature_extractor must return a torch.Tensor.")
            if features.ndim < 2:
                raise ValueError("feature_extractor must return at least two dimensions with batch first.")
            if features.shape[0] != zero_one.shape[0]:
                raise ValueError("feature_extractor changed the batch dimension.")
            if not torch.isfinite(features).all():
                raise ValueError("feature_extractor returned NaN or infinity.")
            stored_features.append(features.detach().flatten(start_dim=1).cpu())
        self.synthetic_images_uint8 = torch.cat(stored_images, dim=0)
        self._synthetic_features = torch.cat(stored_features, dim=0)

    def _fit_surrogate_clusters(self) -> None:
        features = self._synthetic_features.float()
        kmeans = KMeans(
            n_clusters=self.config.clusters,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.random_seed,
            algorithm="lloyd",
        )
        raw_labels = torch.from_numpy(kmeans.fit_predict(features.numpy())).long()
        raw_centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        normalized_features = functional.normalize(features, dim=1)
        normalized_centroids = functional.normalize(raw_centroids, dim=1)

        retained_centroids: list[Tensor] = []
        retained_cohesion: list[float] = []
        for cluster_index in range(self.config.clusters):
            member_mask = raw_labels.eq(cluster_index)
            member_count = int(member_mask.sum())
            if member_count < self.config.min_cluster_size:
                continue
            cohesion = float((normalized_features[member_mask] @ normalized_centroids[cluster_index]).mean())
            if cohesion >= self.config.cohesion_threshold:
                retained_centroids.append(raw_centroids[cluster_index])
                retained_cohesion.append(cohesion)
        if len(retained_centroids) < 2:
            raise ValueError(
                "Fewer than two clusters passed cohesion/min-size filtering; lower cohesion_threshold or min_cluster_size."
            )
        centroids = torch.stack(retained_centroids)
        reassigned = torch.cat(
            [
                torch.cdist(features[start:end], centroids).argmin(dim=1)
                for start, end in batch_ranges(features.shape[0], self.config.synthetic_batch_size)
            ]
        )
        self.synthetic_labels = reassigned.long()
        self.cluster_centroids = centroids
        self.cluster_cohesion = retained_cohesion
        del self._synthetic_features

    def _build_classifier(self) -> None:
        if self.cluster_centroids is None:
            raise RuntimeError("Surrogate clusters were not prepared.")
        num_classes = self.cluster_centroids.shape[0]
        in_channels = self.adapter.image_shape[0]
        if self.classifier is None and self.classifier_factory is not None:
            self.classifier = self.classifier_factory(in_channels, num_classes)
        if self.classifier is None:
            self.classifier = TimeConditionedResNet(
                in_channels=in_channels,
                num_classes=num_classes,
                base_width=self.config.classifier_base_width,
                blocks=self.config.classifier_blocks,
                timestep_embedding_dim=self.config.timestep_embedding_dim,
            )
        self.classifier.to(self.device)
        classifier_dtype = self._classifier_dtype()
        with torch.no_grad():
            probe_images = torch.zeros((2, *self.adapter.image_shape), device=self.device, dtype=classifier_dtype)
            probe_timesteps = self.adapter.classifier_timesteps(torch.zeros(2, dtype=torch.long, device=self.device))
            probe_timesteps = probe_timesteps.to(self.device)
            probe = self.classifier(probe_images, probe_timesteps)
        if probe.shape != (2, num_classes):
            raise ValueError(f"classifier must return shape (batch, {num_classes}); got {tuple(probe.shape)}.")

    def _train_classifier(self) -> None:
        if self.synthetic_images_uint8 is None or self.synthetic_labels is None:
            raise RuntimeError("Synthetic SIDE training data were not prepared.")
        if self.classifier is None:
            raise RuntimeError("The SIDE classifier was not prepared.")
        dataset = TensorDataset(self.synthetic_images_uint8, self.synthetic_labels)
        loader_generator = torch.Generator(device="cpu").manual_seed(self.config.random_seed)
        loader = DataLoader(
            dataset,
            batch_size=self.config.classifier_batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.config.classifier_learning_rate,
            weight_decay=self.config.classifier_weight_decay,
        )
        noise_generator = torch.Generator(device="cpu").manual_seed(self.config.random_seed + 17)
        classifier_dtype = self._classifier_dtype()
        self.classifier.train()
        for _epoch in range(self.config.classifier_epochs):
            loss_sum = 0.0
            sample_count = 0
            for encoded_images, batch_labels in loader:
                clean = decode_uint8(encoded_images, self.config.image_range).to(self.device)
                target_labels = batch_labels.to(self.device)
                timesteps = torch.randint(
                    0,
                    self.adapter.num_timesteps,
                    (clean.shape[0],),
                    generator=noise_generator,
                    device="cpu",
                ).to(self.device)
                noise = torch.randn(clean.shape, generator=noise_generator, device="cpu").to(self.device)
                noisy = self.adapter.q_sample(clean, timesteps, noise)
                noisy = validate_image_batch(noisy, self.adapter.image_shape, expected_count=clean.shape[0])
                classifier_inputs = noisy.to(device=self.device, dtype=classifier_dtype)
                classifier_timesteps = self.adapter.classifier_timesteps(timesteps).to(self.device)
                logits = self.classifier(classifier_inputs, classifier_timesteps)
                loss = functional.cross_entropy(logits, target_labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("SIDE classifier training produced NaN or infinity.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * clean.shape[0]
                sample_count += clean.shape[0]
            if sample_count == 0:
                raise RuntimeError("Synthetic classifier loader produced no samples.")
            self.training_history.append(loss_sum / sample_count)
        self.classifier.eval()

    def _condition_gradient(self, noisy_images: Tensor, timesteps: Tensor, labels: Tensor) -> Tensor:
        if self.classifier is None:
            raise RuntimeError("The SIDE classifier was not prepared.")
        if not noisy_images.is_floating_point():
            raise TypeError("SIDE guidance images must use a floating-point dtype.")
        original_device = noisy_images.device
        original_dtype = noisy_images.dtype
        with torch.enable_grad():
            inputs = noisy_images.detach().to(device=self.device, dtype=self._classifier_dtype()).requires_grad_(True)
            classifier_timesteps = timesteps.to(self.device)
            selected_labels = labels.to(device=self.device, dtype=torch.long)
            logits = self.classifier(inputs, classifier_timesteps)
            selected = functional.log_softmax(logits, dim=1).gather(1, selected_labels.view(-1, 1)).sum()
            gradient = torch.autograd.grad(selected, inputs, create_graph=False, retain_graph=False)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("SIDE classifier guidance produced NaN or infinity.")
        self._guidance_calls += 1
        guided = (gradient * self.config.guidance_scale).to(device=original_device, dtype=original_dtype)
        if not torch.isfinite(guided).all():
            raise RuntimeError("SIDE classifier guidance overflowed on the sampling device or dtype.")
        return guided

    def _classifier_dtype(self) -> torch.dtype:
        """Return the classifier's single floating-point parameter dtype."""
        if self.classifier is None:
            raise RuntimeError("The SIDE classifier was not prepared.")
        state_tensors = list(self.classifier.parameters()) + list(self.classifier.buffers())
        if any(tensor.is_complex() for tensor in state_tensors):
            raise ValueError("SIDE classifier parameters and buffers must not use complex dtypes.")
        dtypes = {
            parameter.dtype
            for parameter in self.classifier.parameters()
            if parameter.is_floating_point()
        }
        dtypes.update(buffer.dtype for buffer in self.classifier.buffers() if buffer.is_floating_point())
        if len(dtypes) > 1:
            raise ValueError("SIDE classifier floating-point parameters must use one dtype.")
        return next(iter(dtypes), torch.float32)

    def _feature_extractor_dtype(self) -> torch.dtype:
        """Return the extractor's single floating-point parameter dtype."""
        state_tensors = list(self.feature_extractor.parameters()) + list(self.feature_extractor.buffers())
        if any(tensor.is_complex() for tensor in state_tensors):
            raise ValueError("SIDE feature-extractor parameters and buffers must not use complex dtypes.")
        dtypes = {tensor.dtype for tensor in state_tensors if tensor.is_floating_point()}
        if len(dtypes) > 1:
            raise ValueError("SIDE feature-extractor floating-point parameters and buffers must use one dtype.")
        return next(iter(dtypes), torch.float32)

    def _new_metric_image_buffer(self) -> list[Tensor] | None:
        """Allocate raw-range batch storage only when reference L2 requires it."""
        if self._references_metric is not None and self.config.image_range == "minus_one_one":
            return []
        return None

    @staticmethod
    def _retain_metric_batch(metric_images: list[Tensor] | None, batch: Tensor) -> None:
        """Retain a raw-range batch when reference metrics requested a buffer."""
        if metric_images is not None:
            metric_images.append(batch)

    def _reference_metric_images(self, images: Tensor, metric_images: list[Tensor] | None) -> Tensor:
        """Select the persisted tensor or construct the required raw-range tensor."""
        if self.config.image_range == "zero_one":
            return images
        if metric_images is None:
            raise RuntimeError("Reference metric images were not retained.")
        return torch.cat(metric_images)

    def _sample_guided_batch(self, batch_size: int, labels: Tensor, seed: int) -> Tensor:
        """Generate one batch and prove that classifier guidance was invoked."""
        guidance_calls_before = self._guidance_calls
        batch = self.adapter.sample_with_classifier_guidance(
            batch_size,
            labels=labels,
            gradient_fn=self._condition_gradient,
            seed=seed,
        )
        if self._guidance_calls <= guidance_calls_before:
            raise RuntimeError("SIDE guided sampling did not invoke the classifier-gradient callback.")
        self._sampling_calls += 1
        return validate_image_batch(batch, self.adapter.image_shape, expected_count=batch_size).detach().cpu()

    def run_attack(self) -> ExtractionResult:
        """Run this prepared attack exactly once."""
        return self._execute_once(self._run_attack)

    def _run_attack(self) -> ExtractionResult:
        """Generate cluster-guided candidates and optionally evaluate them against references."""
        if self.cluster_centroids is None:
            raise RuntimeError("Surrogate clusters were not prepared.")
        generated: list[Tensor] = []
        metric_images = self._new_metric_image_buffer()
        generated_labels: list[Tensor] = []
        label_generator = torch.Generator(device="cpu").manual_seed(self.config.random_seed + 29)
        for batch_index, (start, end) in enumerate(
            batch_ranges(self.config.num_generations, self.config.generation_batch_size)
        ):
            labels = torch.randint(
                0,
                self.cluster_centroids.shape[0],
                (end - start,),
                generator=label_generator,
                device="cpu",
            )
            batch = self._sample_guided_batch(
                end - start,
                labels=labels,
                seed=self.config.random_seed + 1_000_003 + batch_index,
            )
            self._retain_metric_batch(metric_images, batch)
            generated.append(to_zero_one(batch, self.config.image_range))
            generated_labels.append(labels)
        images = torch.cat(generated)
        labels = torch.cat(generated_labels)

        nearest_indices: Tensor | None = None
        nearest_distances: Tensor | None = None
        band_metrics: dict[str, dict[str, float | int]] = {}
        metrics: dict[str, Any] = {
            "images_generated": images.shape[0],
            "retained_clusters": self.cluster_centroids.shape[0],
            "cluster_cohesion": self.cluster_cohesion,
            "classifier_epoch_losses": self.training_history,
            "guidance_scale": self.config.guidance_scale,
        }
        if self._references_metric is not None:
            images_for_l2 = self._reference_metric_images(images, metric_images)
            nearest_indices, nearest_distances = nearest_reference(
                images_for_l2,
                self._references_metric,
                block_size=self.config.distance_block_size,
                device=self.config.distance_device,
            )
            band_metrics = l2_band_scores(
                images_for_l2,
                self._references_metric,
                self.config.l2_bands,
                block_size=self.config.distance_block_size,
                device=self.config.distance_device,
            )
            metrics.update(
                {
                    "nearest_l2_mean": float(nearest_distances.mean()),
                    "nearest_l2_median": float(nearest_distances.median()),
                    "nearest_l2_p95": float(np.percentile(nearest_distances.numpy(), 95)),
                    "l2_band_scores": band_metrics,
                    "l2_coordinate_range": self.config.image_range,
                }
            )
            if self.config.similarity_bands:
                if self.reference_score_fn is None:
                    raise RuntimeError("The configured reference score function was not prepared.")
                if self._references_zero_one is None:
                    raise RuntimeError("Reference images were not prepared for the configured similarity score.")
                metrics["similarity_band_scores"] = pairwise_band_scores(
                    images,
                    self._references_zero_one,
                    self.config.similarity_bands,
                    self.reference_score_fn,
                    block_size=self.config.distance_block_size,
                    device=self.config.distance_device,
                )

        records: list[CandidateRecord] = []
        for image_index, label in enumerate(labels.tolist()):
            record_kwargs: dict[str, Any] = {
                "image_index": image_index,
                "source": f"surrogate_cluster:{label}",
                "metadata": {"surrogate_cluster": label},
            }
            if nearest_indices is not None and nearest_distances is not None:
                record_kwargs.update(
                    {
                        "score": float(nearest_distances[image_index]),
                        "nearest_reference_index": int(nearest_indices[image_index]),
                        "nearest_reference_distance": float(nearest_distances[image_index]),
                    }
                )
            records.append(CandidateRecord(**record_kwargs))

        self._record_trace(
            "run_complete",
            sampling_calls=self._sampling_calls,
            guidance_calls=self._guidance_calls,
            images_generated=int(images.shape[0]),
            candidate_count=len(records),
        )
        return ExtractionResult(
            name="SIDE Extraction Result",
            result_id=self.result_id,
            config=self.config,
            images=images,
            candidates=records,
            metrics=metrics,
            provenance={**self.description(), "audit_fingerprint": self.audit_fingerprint},
            execution_trace=self.execution_trace,
            overwrite=self.config.overwrite_results,
        )
