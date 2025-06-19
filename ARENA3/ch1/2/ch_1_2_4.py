# %%
from contourpy import dechunk_filled
import torch as t
from jaxtyping import Float
from torch import Tensor, nn
import einops
from tqdm import tqdm
from IPython.display import HTML, display

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

import pt21_tests as tests
import pt21_utils as utils
from plotly_utils import imshow, line
from dataclasses import dataclass

from typing import Callable, Literal, Any

MAIN = __name__ == "__main__"
# %%
from toy_model import ToyModel, ToyModelConfig, linear_lr, constant_lr, cosine_decay_lr

NUM_WARMUP_STEPS = 2500
NUM_BATCH_UPDATES = 50_000

WEIGHT_DECAY = 1e-2
LEARNING_RATE = 1e-3

BATCH_SIZES = [3, 5, 6, 8, 10, 15, 30, 50, 100, 200, 500, 1000, 2000]

N_FEATURES = 1000
N_INSTANCES = 5
N_HIDDEN = 2
SPARSITY = 0.99
FEATURE_PROBABILITY = 1 - SPARSITY

features_list = [t.randn(size=(2, 100)) for _ in BATCH_SIZES]
hidden_representations_list = [
    t.randn(size=(2, batch_size)) for batch_size in BATCH_SIZES
]

utils.plot_features_in_2d(
    features_list + hidden_representations_list,
    colors=[["blue"] for _ in range(len(BATCH_SIZES))]
    + [["red"] for _ in range(len(BATCH_SIZES))],
    title="Double Descent & Superposition (num features = 100)",
    subplot_titles=[f"Features (batch={bs})" for bs in BATCH_SIZES]
    + [f"Data (batch={bs})" for bs in BATCH_SIZES],
    allow_different_limits_across_subplots=True,
    n_rows=2,
)


# %%
@dataclass
class ToySAEConfig:
    n_inst: int
    d_in: int
    d_sae: int
    sparsity_coeff: float = 0.2
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    ste_epsilon: float = 0.01


class ToySAE(nn.Module):
    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]

    def __init__(self, cfg: ToySAEConfig, model: ToyModel) -> None:
        """

        - d_sae, which is the number of activations in the SAE's hidden layer (i.e. the latent dimension).
          Note that we want the SAE latents to correspond to the original data features, which is why we'll need d_sae >= n_features (usually we'll have equality in this section).
        - d_in, which is the SAE input dimension. This is the same as d_hidden from the previous sections because the SAE is reconstructing the model's hidden activations, however calling it d_hidden in the context of an SAE would be confusing. Usually in this section, we'll have d_in = d_hidden = 2, so we can visualize the results.

        - n_inst, which means the same as it does in your ToyModel class
        - d_in, the input size to your SAE (equal to d_hidden of your ToyModel class)
        - d_sae, the SAE's latent dimension size
        - sparsity_coeff, which is used in your loss function
        - weight_normalize_eps, which is added to the denominator whenever you normalize weights
        - tied_weights, which is a boolean determining whether your encoder and decoder weights are tied
        - ste_epsilon, which is only relevant for JumpReLU SAEs later on
        """
        super(ToySAE, self).__init__()

        assert (
            cfg.d_in == model.cfg.d_hidden
        ), "Model's hidden dim doesn't match SAE input dim"
        self.cfg = cfg
        self.model = model.requires_grad_(False)
        self.model.W.data[1:] = self.model.W.data[0]
        self.model.b_final.data[1:] = self.model.b_final.data[0]

        # b_enc, b_dec, W_enc and _W_dec
        if cfg.tied_weights:
            self._W_dec = None
        else:
            _W_dec = t.zeros((cfg.n_inst, cfg.d_sae, cfg.d_in))
            self._W_dec = nn.Parameter(t.nn.init.kaiming_uniform_(_W_dec))

        self.b_enc = nn.Parameter(t.zeros((cfg.n_inst, cfg.d_sae)))
        self.b_dec = nn.Parameter(t.zeros((cfg.n_inst, cfg.d_in)))
        W_enc = t.zeros((cfg.n_inst, cfg.d_in, cfg.d_sae))
        self.W_enc = nn.Parameter(t.nn.init.kaiming_uniform_(W_enc))

        self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "inst d_sae d_in"]:
        """
        Returns decoder weights, normalized over the autoencoder input dimension.
        """
        # You'll fill this in later
        norm = t.norm(self.W_dec, dim=-1)

        return (self.W_dec) / (norm[..., t.newaxis] + self.cfg.weight_normalize_eps)

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch inst d_in"]:
        """
        Generates a batch of hidden activations from our model.
        """
        # You'll fill this in later
        x = self.model.generate_batch(batch_size=batch_size)

        return einops.einsum(
            x, self.model.W, "batch inst feat, inst hid feat -> batch inst hid"
        )

    def forward(
        self, h: Float[Tensor, "batch inst d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, "batch inst"],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Forward pass on the autoencoder.

        Args:
            h: hidden layer activations of model

        Returns:
            loss_dict:       dict of different loss terms, each dict value having shape (batch_size, n_inst)
            loss:            total loss (i.e. sum over terms of loss dict), same shape as loss_dict values
            acts_post:       autoencoder latent activations, after applying ReLU
            h_reconstructed: reconstructed autoencoder input
        """
        h_centered = h - self.b_dec
        z_act = einops.einsum(
            self.W_enc,
            h_centered,
            "n_inst d_in d_sae,  batch n_inst d_in->batch n_inst d_sae",
        )
        z = t.relu(z_act + self.b_enc)

        h_rec = (
            einops.einsum(
                self.W_dec_normalized,
                z,
                "n_inst d_sae d_in,  batch n_inst d_sae->batch n_inst d_in",
            )
            + self.b_dec
        )

        loss_rec = t.mean((h - h_rec) ** 2, dim=-1)
        loss_sparse = t.sum(t.abs(z), dim=-1)

        loss_dict = {"L_reconstruction": loss_rec, "L_sparsity": loss_sparse}

        loss = loss_rec + loss_sparse * self.cfg.sparsity_coeff

        return loss_dict, loss, z, h_rec

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        resample_method: Literal["simple", "advanced", None] = None,
        resample_freq: int = 2500,
        resample_window: int = 500,
        resample_scale: float = 0.5,
        hidden_sample_size: int = 256,
    ) -> list[dict[str, Any]]:
        """
        Optimizes the autoencoder using the given hyperparameters.

        Args:
            model:              we reconstruct features from model's hidden activations
            batch_size:         size of batches we pass through model & train autoencoder on
            steps:              number of optimization steps
            log_freq:           number of optimization steps between logging
            lr:                 learning rate
            lr_scale:           learning rate scaling function
            resample_method:    method for resampling dead latents
            resample_freq:      number of optimization steps between resampling dead latents
            resample_window:    number of steps needed for us to classify a neuron as dead
            resample_scale:     scale factor for resampled neurons
            hidden_sample_size: size of hidden value sample we add to the logs (for eventual visualization)

        Returns:
            data_log:           dictionary containing data we'll use for visualization
        """
        assert resample_window <= resample_freq

        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)  # betas=(0.0, 0.999)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists of dicts to store data we'll eventually be plotting
        data_log = []

        for step in progress_bar:
            # Resample dead latents
            if (resample_method is not None) and ((step + 1) % resample_freq == 0):
                frac_active_in_window = t.stack(
                    frac_active_list[-resample_window:], dim=0
                )
                if resample_method == "simple":
                    self.resample_simple(frac_active_in_window, resample_scale)
                elif resample_method == "advanced":
                    self.resample_advanced(
                        frac_active_in_window, resample_scale, batch_size
                    )

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Get a batch of hidden activations from the model
            with t.inference_mode():
                h = self.generate_batch(batch_size)

            # Optimize
            loss_dict, loss, acts, _ = self.forward(h)
            loss.mean(0).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights by modifying them directly (if not using tied weights)
            if not self.cfg.tied_weights:
                self.W_dec.data = self.W_dec_normalized.data

            # Calculate the mean sparsities over batch dim for each feature
            frac_active = (acts.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            # Display progress bar, and log a bunch of values for creating plots / animations
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    lr=step_lr,
                    loss=loss.mean(0).sum().item(),
                    frac_active=frac_active.mean().item(),
                    **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
                )
                with t.inference_mode():
                    loss_dict, loss, acts, h_r = self.forward(
                        h := self.generate_batch(hidden_sample_size)
                    )
                data_log.append(
                    {
                        "steps": step,
                        "frac_active": (acts.abs() > 1e-8)
                        .float()
                        .mean(0)
                        .detach()
                        .cpu(),
                        "loss": loss.detach().cpu(),
                        "h": h.detach().cpu(),
                        "h_r": h_r.detach().cpu(),
                        **{
                            name: param.detach().cpu()
                            for name, param in self.named_parameters()
                        },
                        **{
                            name: loss_term.detach().cpu()
                            for name, loss_term in loss_dict.items()
                        },
                    }
                )

        return data_log

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        """
        Resamples dead latents, by modifying the model's weights and biases inplace.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (d_in,), and normalize these vectors
            - Set new values of W_dec and W_enc to be these normalized vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """

        dead_lats_inst_idx, dead_lats_d_idx = t.where(
            t.sum(frac_active_in_window, dim=0) == 0
        )

        random_v = t.rand(
            dead_lats_inst_idx.shape[0], self.cfg.d_in, device=self._W_dec.device
        )
        random_v = random_v / t.norm(random_v, dim=-1)[:, t.newaxis]

        self._W_dec[dead_lats_inst_idx, dead_lats_d_idx, :] = random_v
        self.W_enc[dead_lats_inst_idx, :, dead_lats_d_idx] = random_v * resample_scale

        self.b_enc[dead_lats_inst_idx, dead_lats_d_idx] = 0

    @t.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        batch_size: int,
    ) -> None:
        """
        Resamples latents that have been dead for `dead_feature_window` steps, according to `frac_active`.

        Resampling method is:
            - Compute the L2 reconstruction loss produced from the hidden state vectors `h`
            - Randomly choose values of `h` with probability proportional to their reconstruction loss
            - Set new values of W_dec and W_enc to be these (centered and normalized) vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        raise NotImplementedError()


# Go back up and edit your `ToySAE.__init__` method, then run the test below

tests.test_sae_init(ToySAE)
tests.test_sae_W_dec_normalized(ToySAE)
tests.test_sae_generate_batch(ToySAE)
tests.test_sae_forward(ToySAE)
tests.test_resample_simple(ToySAE)


# %%

# %%
d_hidden = d_in = 2
n_features = d_sae = 5
n_inst = 16

# Create a toy model, and train it to convergence
cfg = ToyModelConfig(n_inst=n_inst, n_features=n_features, d_hidden=d_hidden)
model = ToyModel(cfg=cfg, device=device, feature_probability=0.025)
model.optimize()

sae = ToySAE(cfg=ToySAEConfig(n_inst=n_inst, d_in=d_in, d_sae=d_sae), model=model)

h = sae.generate_batch(512)

utils.plot_features_in_2d(model.W[:8], title="Base model")
utils.plot_features_in_2d(
    einops.rearrange(h[:, :8], "batch inst d_in -> inst d_in batch"),
    title="Hidden state representation of a random batch of data",
)
# %%
data_log = sae.optimize(steps=20_000)

# %%

utils.animate_features_in_2d(
    data_log,
    instances=list(range(8)),  # only plot the first 8 instances
    rows=["W_enc", "_W_dec"],
    filename=str("animation-training.html"),
    title="SAE on toy model",
)

# If this display code doesn't work, try opening the animation in your browser from where it gets saved
with open("animation-training.html") as f:
    display(HTML(f.read()))
# %%
utils.frac_active_line_plot(
    frac_active=t.stack([data["frac_active"] for data in data_log]),
    title="Probability of sae features being active during training",
    avg_window=20,
)
# %%
resampling_sae = ToySAE(
    cfg=ToySAEConfig(n_inst=n_inst, d_in=d_in, d_sae=d_sae), model=model
)

resampling_data_log = resampling_sae.optimize(steps=20_000, resample_method="simple")

utils.animate_features_in_2d(
    resampling_data_log,
    rows=["W_enc", "_W_dec"],
    instances=list(range(8)),  # only plot the first 8 instances
    filename=str(section_dir / "animation-training-resampling.html"),
    color_resampled_latents=True,
    title="SAE on toy model (with resampling)",
)

utils.frac_active_line_plot(
    frac_active=t.stack([data["frac_active"] for data in resampling_data_log]),
    title="Probability of sae features being active during training",
    avg_window=20,
)
# %%
#################
# Gated units


class GatedToySAE(ToySAE):
    W_gate: Float[Tensor, "inst d_in d_sae"]
    b_gate: Float[Tensor, "inst d_sae"]
    r_mag: Float[Tensor, "inst d_sae"]
    b_mag: Float[Tensor, "inst d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_dec: Float[Tensor, "inst d_in"]

    def __init__(self, cfg: ToySAEConfig, model: ToyModel):
        super(ToySAE, self).__init__()

        assert (
            cfg.d_in == model.cfg.d_hidden
        ), "ToyModel's hidden dim doesn't match SAE input dim"
        self.cfg = cfg
        self.model = model.requires_grad_(False)
        self.model.W.data[1:] = self.model.W.data[0]
        self.model.b_final.data[1:] = self.model.b_final.data[0]

        self._W_dec = (
            None
            if self.cfg.tied_weights
            else nn.Parameter(
                nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_sae, cfg.d_in)))
            )
        )
        self.b_dec = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_in))

        self.W_gate = nn.Parameter(
            nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_in, cfg.d_sae)))
        )
        self.b_gate = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_sae))
        self.r_mag = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_sae))
        self.b_mag = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_sae))

        self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_gate.transpose(-1, -2)

    @property
    def W_mag(self) -> Float[Tensor, "inst d_in d_sae"]:
        return self.r_mag.exp().unsqueeze(1) * self.W_gate

    def forward(
        self, h: Float[Tensor, "batch inst d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, ""],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Same as previous forward function, but allows for gated case as well (in which case we have different
        functional form, as well as a new term "L_aux" in the loss dict).
        """
        h_cent = h - self.b_dec

        # Compute the gating terms (pi_gate(x) and f_gate(x) in the paper)
        gating_pre_activation = (
            einops.einsum(
                h_cent,
                self.W_gate,
                "batch inst d_in, inst d_in d_sae -> batch inst d_sae",
            )
            + self.b_gate
        )
        active_features = (gating_pre_activation > 0).float()

        # Compute the magnitude term (f_mag(x) in the paper)
        magnitude_pre_activation = (
            einops.einsum(
                h_cent,
                self.W_mag,
                "batch inst d_in, inst d_in d_sae -> batch inst d_sae",
            )
            + self.b_mag
        )
        feature_magnitudes = t.relu(magnitude_pre_activation)

        # Compute the hidden activations (f˜(x) in the paper)
        acts_post = active_features * feature_magnitudes

        # Compute reconstructed input
        h_reconstructed = (
            einops.einsum(
                acts_post,
                self.W_dec,
                "batch inst d_sae, inst d_sae d_in -> batch inst d_in",
            )
            + self.b_dec
        )

        # Compute loss terms
        gating_post_activation = t.relu(gating_pre_activation)
        via_gate_reconstruction = (
            einops.einsum(
                gating_post_activation,
                self.W_dec.detach(),
                "batch inst d_sae, inst d_sae d_in -> batch inst d_in",
            )
            + self.b_dec.detach()
        )
        loss_dict = {
            "L_reconstruction": (h_reconstructed - h).pow(2).mean(-1),
            "L_sparsity": gating_post_activation.sum(-1),
            "L_aux": (via_gate_reconstruction - h).pow(2).sum(-1),
        }

        loss = (
            loss_dict["L_reconstruction"]
            + self.cfg.sparsity_coeff * loss_dict["L_sparsity"]
            + loss_dict["L_aux"]
        )

        assert sorted(loss_dict.keys()) == ["L_aux", "L_reconstruction", "L_sparsity"]
        return loss_dict, loss, acts_post, h_reconstructed

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        dead_latents_mask = (frac_active_in_window < 1e-8).all(
            dim=0
        )  # [instances d_sae]
        n_dead = int(dead_latents_mask.int().sum().item())

        replacement_values = t.randn((n_dead, self.cfg.d_in), device=self.W_gate.device)
        replacement_values_normed = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True)
            + self.cfg.weight_normalize_eps
        )

        # New names for weights & biases to resample
        self.W_gate.data.transpose(-1, -2)[dead_latents_mask] = (
            resample_scale * replacement_values_normed
        )
        self.W_dec.data[dead_latents_mask] = replacement_values_normed
        self.b_mag.data[dead_latents_mask] = 0.0
        self.b_gate.data[dead_latents_mask] = 0.0
        self.r_mag.data[dead_latents_mask] = 0.0

    @t.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        batch_size: int,
    ) -> None:
        h = self.generate_batch(batch_size)
        l2_loss = self.forward(h)[0]["L_reconstruction"]

        for instance in range(self.cfg.n_inst):
            is_dead = (frac_active_in_window[:, instance] < 1e-8).all(dim=0)
            dead_latents = t.nonzero(is_dead).squeeze(-1)
            n_dead = dead_latents.numel()
            if n_dead == 0:
                continue

            l2_loss_instance = l2_loss[:, instance]  # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue

            distn = t.distributions.Categorical(
                probs=l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum()
            )
            replacement_indices = distn.sample((n_dead,))  # type: ignore

            replacement_values = (h - self.b_dec)[
                replacement_indices, instance
            ]  # [n_dead d_in]
            replacement_values_normalized = replacement_values / (
                replacement_values.norm(dim=-1, keepdim=True)
                + self.cfg.weight_normalize_eps
            )

            W_gate_norm_alive_mean = (
                self.W_gate[instance, :, ~is_dead].norm(dim=0).mean().item()
                if (~is_dead).any()
                else 1.0
            )

            # New names for weights & biases to resample
            self.W_dec.data[instance, dead_latents, :] = replacement_values_normalized
            self.W_gate.data[instance, :, dead_latents] = (
                replacement_values_normalized.T
                * W_gate_norm_alive_mean
                * resample_scale
            )
            self.b_mag.data[instance, dead_latents] = 0.0
            self.b_gate.data[instance, dead_latents] = 0.0
            self.r_mag.data[instance, dead_latents] = 0.0


# %%
gated_sae = GatedToySAE(
    cfg=ToySAEConfig(
        n_inst=n_inst,
        d_in=d_in,
        d_sae=d_sae,
        sparsity_coeff=1.0,
    ),
    model=model,
)
gated_data_log = gated_sae.optimize(steps=20_000, resample_method="advanced")

# Animate the best instances, ranked according to average loss near the end of training
n_inst_to_plot = 4
n_batches_for_eval = 10
avg_loss = t.concat([d["loss"] for d in gated_data_log[-n_batches_for_eval:]]).mean(0)
best_instances = avg_loss.topk(n_inst_to_plot, largest=False).indices.tolist()

utils.animate_features_in_2d(
    gated_data_log,
    rows=["W_gate", "_W_dec", "h", "h_r"],
    instances=best_instances,
    filename=str("animation-training-gated.html"),
    color_resampled_latents=True,
    title="SAE on toy model",
)


# %%%
class CustomFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: t.Tensor, n: int) -> t.Tensor:
        # Save any necessary information for backward pass
        ctx.save_for_backward(input)
        ctx.n = n  # Save n as it will be needed in the backward pass
        # Compute the output
        return input**n

    @staticmethod
    def backward(ctx: Any, grad_output: t.Tensor) -> tuple[t.Tensor, None]:
        # Retrieve saved tensors and n
        (input,) = ctx.saved_tensors
        n = ctx.n
        # Return gradient for input and None for n (as it's not a Tensor)
        return n * (input ** (n - 1)) * grad_output, None


# Test our function, and its gradient
input = t.tensor(3.0, requires_grad=True)
output = CustomFunction.apply(input, 2)
output.backward()

t.testing.assert_close(output, t.tensor(9.0))
t.testing.assert_close(input.grad, t.tensor(6.0))


# %%
def rectangle(x: Tensor, width: float = 1.0) -> Tensor:
    """
    Returns the rectangle function value, i.e. K(x) = 1[|x| < width/2], as a float.
    """
    return (x.abs() < width / 2).float()


class Heaviside(t.autograd.Function):
    """
    Implementation of the Heaviside step function, using straight through estimators for the derivative.

        forward:
            H(z,θ,ε) = 1[z > θ]

        backward:
            dH/dz := None
            dH/dθ := -1/ε * K(z/ε)

            where K is the rectangle kernel function with width 1, centered at 0: K(u) = 1[|u| < 1/2]
    """

    @staticmethod
    def forward(ctx: Any, z: t.Tensor, theta: t.Tensor, eps: float) -> t.Tensor:
        ctx.save_for_backward(z, theta)
        ctx.eps = eps

        return (z - theta > 0).to(dtype=z.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: t.Tensor) -> tuple[t.Tensor, t.Tensor, None]:
        z, theta = ctx.saved_tensors
        return (
            t.zeros(z.shape, device=z.device),
            -(1 / ctx.eps) * rectangle((z - theta) / ctx.eps) * grad_output,
            None,
        )


# Test our Heaviside function, and its pseudo-gradient
z = t.tensor([[1.0, 1.4, 1.6, 2.0]], requires_grad=True)
theta = t.tensor([1.5, 1.5, 1.5, 1.5], requires_grad=True)
eps = 0.5
output = Heaviside.apply(z, theta, eps)
output.backward(
    t.ones_like(output)
)  # equiv to backprop on each of the 5 elements of z independently

# Test values
t.testing.assert_close(
    output, t.tensor([[0.0, 0.0, 1.0, 1.0]])
)  # expect H(θ,z,ε) = 1[z > θ]
t.testing.assert_close(
    theta.grad, t.tensor([0.0, -2.0, -2.0, 0.0])
)  # expect dH/dθ = -1/ε * K((z-θ)/ε)
t.testing.assert_close(z.grad, t.tensor([[0.0, 0.0, 0.0, 0.0]]))  # expect dH/dz = zero
# assert z.grad is None  # expect dH/dz = None

# Test handling of batch dimension
theta.grad = None
output_stacked = Heaviside.apply(t.concat([z, z]), theta, eps)
output_stacked.backward(t.ones_like(output_stacked))
t.testing.assert_close(theta.grad, 2 * t.tensor([0.0, -2.0, -2.0, 0.0]))

print("All tests for `Heaviside` passed!")
# %%


class JumpReLU(t.autograd.Function):
    """
    Implementation of the JumpReLU function, using straight through estimators for the derivative.

        forward:
            J(z,θ,ε) = z * 1[z > θ]

        backward:
            dJ/dθ := -θ/ε * K((z - θ)/ε)
            dJ/dz := 1[z > θ]

            where K is the rectangle kernel function with width 1, centered at 0: K(u) = 1[|u| < 1/2]
    """

    @staticmethod
    def forward(ctx: Any, z: t.Tensor, theta: t.Tensor, eps: float) -> t.Tensor:
        ctx.save_for_backward(z, theta)
        ctx.eps = eps

        return z * Heaviside.apply(z, theta, eps)

    @staticmethod
    def backward(ctx: Any, grad_output: t.Tensor) -> tuple[t.Tensor, t.Tensor, None]:
        z, theta = ctx.saved_tensors
        dz = Heaviside.apply(z, theta, ctx.eps) * grad_output
        dtheta = -(theta / ctx.eps) * rectangle((z - theta) / ctx.eps) * grad_output
        return dz, dtheta, None


# Test our JumpReLU function, and its pseudo-gradient
z = t.tensor([[1.0, 1.4, 1.6, 2.0]], requires_grad=True)
theta = t.tensor([1.5, 1.5, 1.5, 1.5], requires_grad=True)
eps = 0.5
output = JumpReLU.apply(z, theta, eps)

output.backward(
    t.ones_like(output)
)  # equiv to backprop on each of the 5 elements of z independently

# Test values
t.testing.assert_close(
    output, t.tensor([[0.0, 0.0, 1.6, 2.0]])
)  # expect J(θ,z,ε) = z * 1[z > θ]
t.testing.assert_close(
    theta.grad, t.tensor([0.0, -3.0, -3.0, 0.0])
)  # expect dJ/dθ = -θ/ε * K((z-θ)/ε)
t.testing.assert_close(
    z.grad, t.tensor([[0.0, 0.0, 1.0, 1.0]])
)  # expect dJ/dz = 1[z > θ]

print("All tests for `JumpReLU` passed!")

# %%

THETA_INIT = 0.1


class JumpReLUToySAE(ToySAE):
    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]
    log_theta: Float[Tensor, "inst d_sae"]

    def __init__(self, cfg: ToySAEConfig, model: ToyModel):
        super(ToySAE, self).__init__()

        assert (
            cfg.d_in == model.cfg.d_hidden
        ), "ToyModel's hidden dim doesn't match SAE input dim"
        self.cfg = cfg
        self.model = model.requires_grad_(False)
        self.model.W.data[1:] = self.model.W.data[0]
        self.model.b_final.data[1:] = self.model.b_final.data[0]

        self._W_dec = (
            None
            if self.cfg.tied_weights
            else nn.Parameter(
                nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_sae, cfg.d_in)))
            )
        )
        self.b_dec = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_in))

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_in, cfg.d_sae)))
        )
        self.b_enc = nn.Parameter(t.zeros(cfg.n_inst, cfg.d_sae))
        self.log_theta = nn.Parameter(
            t.full((cfg.n_inst, cfg.d_sae), t.log(t.tensor(THETA_INIT)))
        )

        self.to(device)

    @property
    def theta(self) -> Float[Tensor, "inst d_sae"]:
        return self.log_theta.exp()

    def forward(
        self, h: Float[Tensor, "batch inst d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, ""],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Same as previous forward function, but allows for gated case as well (in which case we have different
        functional form, as well as a new term "L_aux" in the loss dict).
        """
        h_cent = h - self.b_dec

        acts_pre = (
            einops.einsum(
                h_cent,
                self.W_enc,
                "batch inst d_in, inst d_in d_sae -> batch inst d_sae",
            )
            + self.b_enc
        )
        # print(self.theta.mean(), self.theta.std(), self.theta.min(), self.theta.max())
        acts_relu = t.relu(acts_pre)
        acts_post = JumpReLU.apply(acts_relu, self.theta, self.cfg.ste_epsilon)

        h_reconstructed = (
            einops.einsum(
                acts_post,
                self.W_dec,
                "batch inst d_sae, inst d_sae d_in -> batch inst d_in",
            )
            + self.b_dec
        )

        loss_dict = {
            "L_reconstruction": (h_reconstructed - h).pow(2).mean(-1),
            "L_sparsity": Heaviside.apply(
                acts_relu, self.theta, self.cfg.ste_epsilon
            ).sum(-1),
        }

        loss = (
            loss_dict["L_reconstruction"]
            + self.cfg.sparsity_coeff * loss_dict["L_sparsity"]
        )

        return loss_dict, loss, acts_post, h_reconstructed

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        dead_latents_mask = (frac_active_in_window < 1e-8).all(
            dim=0
        )  # [instances d_sae]
        n_dead = int(dead_latents_mask.int().sum().item())

        replacement_values = t.randn((n_dead, self.cfg.d_in), device=self.W_enc.device)
        replacement_values_normed = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True)
            + self.cfg.weight_normalize_eps
        )

        # New names for weights & biases to resample
        self.W_enc.data.transpose(-1, -2)[dead_latents_mask] = (
            resample_scale * replacement_values_normed
        )
        self.W_dec.data[dead_latents_mask] = replacement_values_normed
        self.b_enc.data[dead_latents_mask] = 0.0
        self.log_theta.data[dead_latents_mask] = t.log(t.tensor(THETA_INIT))

    @t.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        batch_size: int,
    ) -> None:
        h = self.generate_batch(batch_size)
        l2_loss = self.forward(h)[0]["L_reconstruction"]

        for instance in range(self.cfg.n_inst):
            is_dead = (frac_active_in_window[:, instance] < 1e-8).all(dim=0)
            dead_latents = t.nonzero(is_dead).squeeze(-1)
            n_dead = dead_latents.numel()
            if n_dead == 0:
                continue

            l2_loss_instance = l2_loss[:, instance]  # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue

            distn = t.distributions.Categorical(
                probs=l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum()
            )
            replacement_indices = distn.sample((n_dead,))  # type: ignore

            replacement_values = (h - self.b_dec)[
                replacement_indices, instance
            ]  # [n_dead d_in]
            replacement_values_normalized = replacement_values / (
                replacement_values.norm(dim=-1, keepdim=True)
                + self.cfg.weight_normalize_eps
            )

            W_enc_norm_alive_mean = (
                self.W_enc[instance, :, ~is_dead].norm(dim=0).mean().item()
                if (~is_dead).any()
                else 1.0
            )

            # New names for weights & biases to resample
            self.b_enc.data[instance, dead_latents] = 0.0
            self.log_theta.data[instance, dead_latents] = t.log(t.tensor(THETA_INIT))
            self.W_dec.data[instance, dead_latents, :] = replacement_values_normalized
            self.W_enc.data[instance, :, dead_latents] = (
                replacement_values_normalized.T * W_enc_norm_alive_mean * resample_scale
            )


jumprelu_sae = JumpReLUToySAE(
    cfg=ToySAEConfig(
        n_inst=n_inst, d_in=d_in, d_sae=d_sae, tied_weights=True, sparsity_coeff=0.1
    ),
    model=model,
)
jumprelu_data_log = jumprelu_sae.optimize(
    steps=20_000, resample_method="advanced"
)  # batch_size=4096?

# Animate the best instances, ranked according to average loss near the end of training
n_inst_to_plot = 4
n_batches_for_eval = 10
avg_loss = t.concat([d["loss"] for d in jumprelu_data_log[-n_batches_for_eval:]]).mean(
    0
)
best_instances = avg_loss.topk(n_inst_to_plot, largest=False).indices.tolist()
# %%
utils.animate_features_in_2d(
    jumprelu_data_log,
    rows=["W_enc", "h", "h_r"],
    instances=best_instances,
    filename=str("animation-training-jumprelu.html"),
    color_resampled_latents=True,
    title="JumpReLU SAE on toy model",
)
# %%
