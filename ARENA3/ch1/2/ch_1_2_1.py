# %%
import torch as t
from jaxtyping import Float
from torch import Tensor, nn

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part31_superposition_and_saes"
#vroot_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
#exercises_dir = root_dir / chapter / "exercises"
# section_dir = exercises_dir / section
#if str(exercises_dir) not in sys.path:
#    sys.path.append(str(exercises_dir))

import pt21_tests as tests
import pt21_utils as utils
from plotly_utils import imshow, line

MAIN = __name__ == "__main__"
# %%
from toy_model import ToyModel, ToyModelConfig

class NeuronModel(ToyModel):
    def forward(self, features: Float[Tensor, "... inst feats"]) -> Float[Tensor, "... inst feats"]:
        h_input = einops.einsum(
            self.W, features, "... inst d_hid d_feat, ... inst d_feat -> ... inst d_hid")
        nonlin_h = t.relu(h_input)
        h_out = einops.einsum(
            self.W, nonlin_h, "... inst d_hid d_feat, ... inst d_hid -> ... inst d_feat")
        return t.relu(h_out + self.b_final)


tests.test_neuron_model(NeuronModel)

# %%
cfg = ToyModelConfig(n_inst=7, n_features=10, d_hidden=5)

importance = 0.75 ** t.arange(1, 1 + cfg.n_features)
feature_probability = t.tensor([0.75, 0.35, 0.15, 0.1, 0.06, 0.02, 0.01])

model = NeuronModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize(steps=10_000)


utils.plot_features_in_Nd(
    model.W,
    height=600,
    width=1000,
    title=f"Neuron model: {cfg.n_features=}, {cfg.d_hidden=}, I<sub>i</sub> = 0.75<sup>i</sup>",
    subplot_titles=[f"1 - S = {i:.2f}" for i in feature_probability.squeeze()],
    neuron_plot=True,
)
# %%
## Computation in superposition

class NeuronComputationModel(ToyModel):
    W1: Float[Tensor, "inst d_hidden feats"]
    W2: Float[Tensor, "inst feats d_hidden"]
    b_final: Float[Tensor, "inst feats"]

    def __init__(
        self,
        cfg: ToyModelConfig,
        feature_probability: float | Tensor = 1.0,
        importance: float | Tensor = 1.0,
        device=device,
    ):
        super(ToyModel, self).__init__()
        self.cfg = cfg

        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_inst, cfg.n_features))
        if isinstance(importance, float):
            importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_inst, cfg.n_features))

        self.W1 = nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_hidden, cfg.n_features))))
        self.W2 = nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.n_features, cfg.d_hidden))))
        self.b_final = nn.Parameter(t.zeros((cfg.n_inst, cfg.n_features)))
        self.to(device)

    def forward(
        self,
        features: Float[Tensor, "... inst feats"],
    ) -> Float[Tensor, "... inst feats"]:
        """
        Performs a single forward pass. For a single instance, this is given by:
            x -> ReLU(W.T @ W @ x + b_final)
        """
        h_1 = einops.einsum(
            self.W1, features, "... inst d_hid d_feat, ... inst d_feat -> ... inst d_hid")
        nonlin_h = t.relu(h_1)
        h_2 = einops.einsum(
            self.W2, nonlin_h, "... inst d_feat d_hid, ... inst d_hid -> ... inst d_feat")
        return t.relu(h_2 + self.b_final)

    # Solution with correlations:
    def generate_batch(self, batch_size) -> Float[Tensor, "batch inst feats"]:
        """
        Generates a batch of data, with optional correlated & anticorrelated features.
        """
        instances = self.cfg.n_inst
        features = self.cfg.n_features
        full_shape = batch_size, instances, features
        features_extraction = t.rand(full_shape, device=self.W1.device)
        features_mag = t.rand(full_shape, device=self.W1.device)*2 - 1
        batch = (features_extraction <= self.feature_probability) * features_mag
        return batch

    def calculate_loss(
        self,
        out: Float[Tensor, "batch inst feats"],
        batch: Float[Tensor, "batch inst feats"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch (as a scalar tensor), using this loss described in the
        Toy Models of Superposition paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `self.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        """

        diff = ((out - abs(batch)) ** 2) * self.importance

        rme = einops.reduce(diff, "batch inst feats -> inst", "mean")
        return t.sum(rme)


tests.test_neuron_computation_model(NeuronComputationModel)
# %%
cfg = ToyModelConfig(n_inst=7, n_features=100, d_hidden=40)

importance = 0.8 ** t.arange(1, 1 + cfg.n_features)
feature_probability = t.tensor([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001])

model = NeuronComputationModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize()


utils.plot_features_in_Nd(
    model.W1,
    height=800,
    width=1600,
    title=f"Neuron computation model: n_features = {cfg.n_features}, d_hidden = {cfg.d_hidden}, I<sub>i</sub> = 0.75<sup>i</sup>",
    subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
    neuron_plot=True,
)
# %%
cfg = ToyModelConfig(n_inst=6, n_features=20, d_hidden=10)

importance = 0.8 ** t.arange(1, 1 + cfg.n_features)
feature_probability = 0.5

model = NeuronComputationModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability,
)
model.optimize()


utils.plot_features_in_Nd_discrete(
    W1=model.W1,
    W2=model.W2,
    title="Neuron computation model (colored discretely, by feature)",
    legend_names=[f"I<sub>{i}</sub> = {importance.squeeze()[i]:.3f}" for i in range(cfg.n_features)],
)
# %%
