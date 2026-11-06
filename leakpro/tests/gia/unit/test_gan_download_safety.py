#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Downloaded GAN weights are loaded as plain tensors, never unpickled as arbitrary objects.

The download is replaced by a local file, so the tests need no network.
"""
from __future__ import annotations

import collections
import shutil
from pathlib import Path

import pytest
import torch

from leakpro.fl_utils import gan_handler


class _Payload:
    """Stands in for an arbitrary pickled object an untrusted checkpoint could carry."""


def _serve(monkeypatch: pytest.MonkeyPatch, source: Path) -> list[str]:
    urls: list[str] = []

    def fake_urlretrieve(url: str, filename: str) -> None:
        urls.append(url)
        shutil.copyfile(source, filename)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    return urls


def _csinva_state_dict(nc: int) -> collections.OrderedDict:
    state = gan_handler._build_dcgan_generator(nz=100, nc=nc, ngf=64).state_dict()
    return collections.OrderedDict((f"main.{k}", v) for k, v in state.items())


def test_github_dcgan_loads_from_a_state_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "netG.pth"
    torch.save(_csinva_state_dict(nc=3), path)
    urls = _serve(monkeypatch, path)
    gen = gan_handler.load_pretrained_gan(github_model="csinva/cifar10_dcgan")
    assert urls == ["https://raw.githubusercontent.com/csinva/gan-vae-pretrained-pytorch/master/cifar10_dcgan/weights/netG_epoch_199.pth"]
    assert gen(torch.randn(2, 100)).shape == (2, 3, 32, 32)


def test_github_download_refuses_pickled_objects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "netG.pth"
    torch.save({"main.0.weight": _Payload()}, path)
    _serve(monkeypatch, path)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|global"):
        gan_handler.load_pretrained_gan(github_model="csinva/cifar10_dcgan")


def test_unknown_github_model_raises_before_downloading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    urls = _serve(monkeypatch, tmp_path / "missing")
    with pytest.raises(ValueError, match="cifar100_dcgan"):
        gan_handler.load_pretrained_gan(github_model="csinva/cifar100_dcgan")
    assert urls == []


def test_stylegan2_checkpoint_needs_explicit_trust(tmp_path: Path) -> None:
    path = tmp_path / "g.pt"
    torch.save({"g_ema": torch.nn.Linear(2, 2)}, path)
    with pytest.raises(ValueError, match="trust_checkpoint"):
        gan_handler.load_pretrained_gan(checkpoint_path=path, architecture="stylegan2")
    gen = gan_handler.load_pretrained_gan(checkpoint_path=path, architecture="stylegan2", trust_checkpoint=True)
    assert isinstance(gen, torch.nn.Linear)
