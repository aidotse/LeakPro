"""Small image-provider doubles for attack orchestration tests."""

from types import SimpleNamespace


def image_sampler(image_shape, sample_fn, *, model=None, num_timesteps=0, q_sample_fn=None, guided_sample_fn=None):
    """Supply controlled images while leaving attack validation and algorithms real."""
    def sample(batch_size, *, conditions, seed):
        return sample_fn(batch_size, conditions, seed)

    def guided(batch_size, *, labels, gradient_fn, seed):
        return guided_sample_fn(batch_size, labels, gradient_fn, seed)

    return SimpleNamespace(
        image_shape=image_shape, model=model, num_timesteps=num_timesteps,
        sample=sample, q_sample=q_sample_fn, classifier_timesteps=lambda timesteps: timesteps,
        sample_with_classifier_guidance=guided if guided_sample_fn is not None else None,
    )
