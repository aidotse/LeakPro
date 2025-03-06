"""Loss functions for GANs. Copied from https://github.com/LetheSec/PLG-MI-Attack."""
import torch
import torch.nn.functional as F  # noqa: N812

AVAILABLE_LOSSES = ["hinge", "dcgan"]


def max_margin_loss(out: torch.Tensor, iden: torch.Tensor) -> torch.Tensor:
    """Compute the max margin loss.

    Args:
        out (torch.Tensor): The output tensor.
        iden (torch.Tensor): The identity tensor.

    Returns:
        torch.Tensor: The computed loss.

    """
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)

    return (-1 * real).mean() + margin.mean()


def cross_entropy_loss(out: torch.Tensor, iden: torch.Tensor) -> torch.Tensor:
    """Computes the cross-entropy loss between the output and the target.

    Args:
        out (torch.Tensor): The output tensor from the model.
        iden (torch.Tensor): The target tensor with the true labels.

    Returns:
        torch.Tensor: The computed cross-entropy loss.

    """
    return torch.nn.CrossEntropyLoss()(out, iden)


def poincare_loss(outputs: torch.Tensor, targets: torch.Tensor, xi: float = 1e-4) -> torch.Tensor:
    """Compute the Poincare loss.

    Args:
        outputs (torch.Tensor): The output tensor from the model.
        targets (torch.Tensor): The target tensor with the true labels.
        xi (float, optional): A small constant to avoid numerical issues. Defaults to 1e-4.

    Returns:
        torch.Tensor: The computed Poincare loss.

    """
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(torch.eye(outputs.shape[-1])[targets] - xi, 0, 1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1) ** 2
    v_norm_squared = torch.norm(v, p=2, dim=1) ** 2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1) ** 2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                     (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss.mean()


def dis_hinge(dis_fake: torch.Tensor, dis_real: torch.Tensor) -> torch.Tensor:
    """Compute the hinge loss for the discriminator.

    Args:
        dis_fake (torch.Tensor): The discriminator output for fake data.
        dis_real (torch.Tensor): The discriminator output for real data.

    Returns:
        torch.Tensor: The computed hinge loss.

    """
    return torch.mean(F.relu(1. - dis_real)) + \
           torch.mean(F.relu(1. + dis_fake))


def gen_hinge(dis_fake: torch.Tensor, dis_real: torch.Tensor = None) -> torch.Tensor:  # noqa: ARG001
    """Compute the hinge loss for the generator.

    Args:
        dis_fake (torch.Tensor): The discriminator output for fake data.
        dis_real (torch.Tensor, optional): The discriminator output for real data. Defaults to None.

    Returns:
        torch.Tensor: The computed hinge loss.

    """
    return -torch.mean(dis_fake)


def dis_dcgan(dis_fake: torch.Tensor, dis_real: torch.Tensor) -> torch.Tensor:
    """Compute the DCGAN loss for the discriminator.

    Args:
        dis_fake (torch.Tensor): The discriminator output for fake data.
        dis_real (torch.Tensor): The discriminator output for real data.

    Returns:
        torch.Tensor: The computed DCGAN loss.

    """
    return torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))


def gen_dcgan(dis_fake: torch.Tensor, dis_real: torch.Tensor = None) -> torch.Tensor:  # noqa: ARG001
    """Compute the DCGAN loss for the generator.

    Args:
        dis_fake (torch.Tensor): The discriminator output for fake data.
        dis_real (torch.Tensor, optional): The discriminator output for real data. Defaults to None.

    Returns:
        torch.Tensor: The computed DCGAN loss.

    """
    return torch.mean(F.softplus(-dis_fake))


class _Loss(object):
    """GAN Loss base class.

    Args:
        loss_type (str)
        is_relativistic (bool)

    """

    def __init__(self, loss_type: str, is_relativistic: bool = False) -> None:
        assert loss_type in AVAILABLE_LOSSES, "Invalid loss. Choose from {}".format(AVAILABLE_LOSSES)
        self.loss_type = loss_type
        self.is_relativistic = is_relativistic

    def _preprocess(self, dis_fake: torch.Tensor, dis_real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_xf_tilde = torch.mean(dis_fake, dim=0, keepdim=True).expand_as(dis_fake)
        c_xr_tilde = torch.mean(dis_real, dim=0, keepdim=True).expand_as(dis_real)
        return dis_fake - c_xr_tilde, dis_real - c_xf_tilde


class DisLoss(_Loss):
    """Discriminator Loss."""

    def __call__(self, dis_fake: torch.Tensor, dis_real: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ANN003, ARG002
        """Compute the loss.

        Args:
            dis_fake (torch.Tensor): The discriminator output for fake data.
            dis_real (torch.Tensor): The discriminator output for real data.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss.

        """
        if not self.is_relativistic:
            if self.loss_type == "hinge":
                return dis_hinge(dis_fake, dis_real)
            if self.loss_type == "dcgan":
                return dis_dcgan(dis_fake, dis_real)
            return None
        d_xf, d_xr = self._preprocess(dis_fake, dis_real)
        if self.loss_type == "hinge":
            return dis_hinge(d_xf, d_xr)
        if self.loss_type == "dcgan":
            d_xf = torch.sigmoid(d_xf)
            d_xr = torch.sigmoid(d_xr)
            return -torch.log(d_xr) - torch.log(1.0 - d_xf)
        raise NotImplementedError


class GenLoss(_Loss):
    """Generator Loss."""

    def __call__(self, dis_fake: torch.Tensor, dis_real: torch.Tensor = None, **kwargs) -> torch.Tensor:  # noqa: ANN003, ARG002
        """Compute the loss.

        Args:
            dis_fake (torch.Tensor): The discriminator output for fake data.
            dis_real (torch.Tensor, optional): The discriminator output for real data. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The computed loss.

        """
        if not self.is_relativistic:
            if self.loss_type == "hinge":
                return gen_hinge(dis_fake, dis_real)
            if self.loss_type == "dcgan":
                return gen_dcgan(dis_fake, dis_real)
            return None
        assert dis_real is not None, "Relativistic Generator loss requires `dis_real`."
        d_xf, d_xr = self._preprocess(dis_fake, dis_real)
        if self.loss_type == "hinge":
            return dis_hinge(d_xr, d_xf)
        if self.loss_type == "dcgan":
            d_xf = torch.sigmoid(d_xf)  # noqa: N806
            d_xr = torch.sigmoid(d_xr)
            return -torch.log(d_xf) - torch.log(1.0 - d_xr)
        raise NotImplementedError
