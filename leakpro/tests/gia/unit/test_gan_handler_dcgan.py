#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""The csinva-style DCGAN built by the handler accepts the framework's flat latent (it natively wants [N, nz, 1, 1])."""

import pytest
import torch

from leakpro.fl_utils.gan_handler import _build_dcgan_generator, load_pretrained_gan
from leakpro.fl_utils.gan_models import LayeredGANWrapper


def test_flat_and_4d_latents_agree() -> None:
    gen = _build_dcgan_generator(nz=100, nc=3, ngf=4).eval()
    z = torch.randn(4, 100)
    flat, spatial = gen(z), gen(z[:, :, None, None])
    assert flat.shape == (4, 3, 32, 32)
    assert torch.allclose(flat, spatial)


def test_state_dict_keys_unchanged_and_layered_wrapper_works() -> None:
    gen = _build_dcgan_generator(nz=100, nc=3, ngf=4).eval()
    assert "0.weight" in gen.state_dict()  # checkpoint keys stay loadable
    z = torch.randn(2, 100)
    assert torch.allclose(LayeredGANWrapper(gen)(z), gen(z))


def test_local_dcgan_checkpoint_round_trips(tmp_path) -> None:
    """A local copy of the csinva weights (with or without the ``main.`` prefix) loads via architecture="dcgan"."""
    gen = _build_dcgan_generator(nz=100, nc=3, ngf=64).eval()
    z = torch.randn(2, 100)
    for prefix in ("", "main."):
        path = tmp_path / f"netG{prefix}.pth"
        torch.save({prefix + k: v for k, v in gen.state_dict().items()}, path)
        loaded = load_pretrained_gan(checkpoint_path=path, architecture="dcgan")
        assert loaded(z).shape == (2, 3, 32, 32)
        assert torch.allclose(loaded(z), gen(z))


def test_github_dcgan_rejects_another_latent_size() -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        load_pretrained_gan(github_model="csinva/cifar10_dcgan", latent_dim=64)
