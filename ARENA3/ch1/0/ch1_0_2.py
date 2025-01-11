# %%
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from aiohttp import DataQueue
import circuitsvis as cv
import datasets
import einops
import einops.layers
import einops.layers.torch
from matplotlib import axes
import numpy as np
import torch as t
import torch
import torch.nn as nn
import wandb
from IPython.display import display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_transformer_from_scratch"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import solutions as solutions
import tests as tests

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda" if t.cuda.is_available() else "cpu"
)

MAIN = __name__ == "__main__"

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device,
)

# %%
# Building blocks for transformer:


@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:

        means = t.mean(residual, axis=-1, keepdim=True)
        vars = t.var(residual, axis=-1, keepdim=True, unbiased=False)

        normalized = (residual - means) / t.sqrt(vars + self.cfg.layer_norm_eps)
        scaled = normalized * self.w + self.b

        return scaled


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens, :]


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:

        indices = t.stack([t.arange(tokens.shape[1]) for _ in range(tokens.shape[0])])

        return self.W_pos[indices, :]


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer(
            "IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device)
        )

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        return attn_out

    def forward_mine(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Compute Q, K and V activations:
        q_activations = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_Q
        )
        k_activations = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_K
        )
        v_activations = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head",
            )
            + self.b_V
        )

        # Get attention matrix:
        attention_logits = einops.einsum(
            q_activations,
            k_activations,
            "batch posq n_heads d_head, batch posk n_heads d_head -> batch n_heads posq posk",
        )

        normed_masked_attention_logits = self.apply_causal_mask(
            attention_logits
            / t.sqrt(t.tensor(cfg.d_head, dtype=t.float32, device=device))
        )
        attention_p = t.softmax(normed_masked_attention_logits, dim=3)

        z = einops.einsum(
            v_activations,
            attention_p,
            "batch posk n_heads d_head, batch n_heads posq posk -> batch posq n_heads d_head",
        )

        result = (
            einops.einsum(
                z,
                self.W_O,
                "batch posq n_heads d_head, n_heads d_head d_model -> batch posq d_model",
            )
            + self.b_O
        )

        return result

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        _, _, n_q, n_k = attn_scores.shape

        all_ones = t.ones(n_q, n_k, device=device, dtype=bool)
        mask = t.triu(all_ones, diagonal=1)

        # masked fill works with broadcasting
        return t.masked_fill(attn_scores, mask, self.IGNORE)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        h_input = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp",
            )
            + self.b_in
        )

        h_activation = gelu_new(h_input)

        return (
            einops.einsum(
                h_activation,
                self.W_out,
                "batch posn d_mlp, d_mlp d_model -> batch posn d_model",
            )
            + self.b_out
        )


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        post_attn = self.attn(self.ln1(resid_pre)) + resid_pre
        post_mlp = self.mlp(self.ln2(post_attn))

        return post_attn + post_mlp


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)

        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:

        return (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch position d_model, d_model d_vocab -> batch position d_vocab",
            )
            + self.b_U
        )


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        embedded = self.embed(tokens)
        pos_embedded = self.pos_embed(tokens)

        transformer_input = embedded + pos_embedded
        for trasf_block in self.blocks:
            transformer_input = trasf_block(transformer_input)

        return self.unembed(self.ln_final(transformer_input))


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 16
    epochs: int = 20
    max_steps_per_epoch: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-2
    use_wandb: bool = True
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None


cfg = Config()
cfg_trainer = TransformerTrainingArgs()

dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns(
    "meta"
)

tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=4,
)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(
    dataset_dict["train"],
    batch_size=cfg_trainer.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset_dict["test"],
    batch_size=cfg_trainer.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


## for predictions:
def sampling_fn(model: DemoTransformer, prompt: str) -> str:
    sampler = solutions.TransformerSampler(model, reference_gpt2.tokenizer)
    output = sampler.sample(
        prompt, temperature=0.7, top_p=0.95, max_tokens_generated=16
    )
    return output


class TransformerTrainer:
    def __init__(
        self, args: TransformerTrainingArgs, model: DemoTransformer, prompt_list=None
    ):
        super().__init__()
        self.model = model
        self.args = args

        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0

        n_workers = 0
        self.train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
        )

        self.prompt_list = prompt_list

    def training_step(
        self, batch: dict[str, Int[Tensor, "batch seq"]]
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        self.optimizer.zero_grad()

        tokens = batch["tokens"].to(device)
        demo_logits = self.model(tokens)
        log_softmax = demo_logits.log_softmax(dim=-1)

        gathered_probs = log_softmax.gather(2, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

        loss = -gathered_probs.mean()

        loss.backward()
        self.optimizer.step()

        self.step += 1

        wandb.log(dict(loss=loss), step=self.step)

        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        n_matches = 0
        total_n = 0

        for batch in self.test_loader:
            tokens = batch["tokens"].to(device)
            logits = model(tokens)
            prediction = logits.argmax(2)

            n_matches += (prediction[:, :-1] == tokens[:, 1:]).sum()
            total_n += prediction[:, :-1].numel()

        accuracy = n_matches / total_n

        wandb.log(dict(accuracy=accuracy), step=self.step)

        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        if self.args.use_wandb:
            full_config = self.args.__dict__.copy()
            full_config.update(self.model.cfg.__dict__)
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_name,
                config=full_config,
            )

        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        inferences_list = []
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):

                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()

            if self.prompt_list is not None:
                sampled = [
                    sampling_fn(model, prompt=prompt) for prompt in self.prompt_list
                ]

                inferences_list.append(
                    [
                        self.step,
                        epoch,
                    ]
                    + sampled
                )
                inf_table = wandb.Table(
                    columns=[
                        "step",
                        "epoch",
                    ]
                    + [f"text{i}" for i in range(len(sampled))],
                    data=inferences_list,
                )
                wandb.log(data=dict(table=inf_table))
        if self.args.use_wandb:
            wandb.finish()


# %%
t.set_grad_enabled(False)  # gradients are not necessary for sampling

model = DemoTransformer(cfg).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False)
tokenizer = reference_gpt2.tokenizer
# %%


class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs):
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an end-of-sequence token. kwargs are
        passed to sample_next_token, to give detailed instructions on how new tokens are chosen.
        """
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]

        for _ in range(max_tokens_generated):
            logits = model(token_ids.unsqueeze(0))
            next_token = self.sample_next_token(token_ids, logits[0, -1, :], **kwargs)

            if next_token == self.tokenizer.eos_token_id:
                break

            token_ids = t.concatenate([token_ids, t.tensor([next_token])])

            output = self.tokenizer.decode(token_ids)
            if verbose:
                print(output)

        return output

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ):
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (
            top_p != 0 and top_k != 0
        ), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        return logits.argmax().item()

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        freq_penalty: float,
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        counts = t.bincount(input_ids, minlength=len(logits)).to(device)

        return logits - freq_penalty * counts

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        dist = t.distributions.categorical.Categorical(logits=logits)
        return dist.sample().item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        top_k_logits, top_k_token_ids = logits.topk(k)

        new_id = t.distributions.categorical.Categorical(logits=top_k_logits).sample()
        return top_k_token_ids[new_id].item()

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        probs = logits.softmax(-1)
        sorted_probs, sorted_ids = t.sort(probs, descending=True)

        prob_cumsum = t.cumsum(sorted_probs, 0)

        n_to_take = max((prob_cumsum < top_p).sum().item() + 1, min_tokens_to_keep)
        filtered_probs = sorted_probs[:n_to_take]
        filtered_ids = sorted_ids[:n_to_take]

        new_id = t.distributions.categorical.Categorical(probs=filtered_probs).sample()

        return filtered_ids[new_id].item()

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int | None = None,
    ) -> list[tuple[float, str]]:
        """
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
        prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
        we've generated `num_returns_sequences` terminating sequences.
        """
        return NotImplementedError()


@dataclass
class Beams:
    """Class to store beams during beam search."""

    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def __getitem__(self, batch_idx) -> "Beams":
        """Allows you to create new beams from old beams by slicing along batch dim (useful for `filter`)."""
        return Beams(
            self.model,
            self.tokenizer,
            self.logprob_sums[batch_idx],
            self.tokens[batch_idx],
        )

    @property
    def logprobs_and_completions(self) -> list[tuple[float, str]]:
        """Returns self as a list of logprob sums and completions (useful for getting final output)."""
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def generate(self, k: int, no_repeat_ngram_size: int | None = None) -> "Beams":
        """
        Starting from the current set of beams (i.e. self.tokens) and returns a new set of `len(self.tokens) * k` beams,
        containing the best `k` continuations for each of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with a repeating n-gram
        of this length.
        """

        # token_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
        batch, seq = self.tokens.shape
        new_tokens = t.empty(batch * k, seq + 1, dtype=self.tokens.dtype)
        new_logprob_sums = t.empty(batch * k, dtype=self.logprob_sums.dtype)

        all_logits = self.model(self.tokens)
        all_logprobs = all_logits.log_softmax(-1)

        for i in range(batch):

            top_k_log_probs, top_k_token_ids = all_logprobs[i, -1, :].topk(k, dim=-1)

            new_tokens[i * k : (i + 1) * k, :-1] = self.tokens[i, :]
            new_tokens[i * k : (i + 1) * k, -1] = top_k_token_ids

            new_logprob_sums[i * k : (i + 1) * k] = (
                self.logprob_sums[i] + top_k_log_probs
            )

        return Beams(self.model, self.tokenizer, new_logprob_sums, new_tokens)

    def filter(self, k: int) -> tuple["Beams", "Beams"]:
        """
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `k` which are also not terminated.
            early_terminations: Beams
                filtered version of self, containing all best `k` which are also terminated.
        """
        best_beams_tokens = []
        best_beams_probs = []

        early_terminations_tokens = []
        early_terminations_probs = []

        _, logprobs_sorted_idxs = self.logprob_sums.sort(descending=True)

        for idx in logprobs_sorted_idxs:
            tokens = self.tokens[idx]
            log_probs = self.logprob_sums[idx]

            if (
                len(early_terminations_tokens) < k
                and tokens[-1] == self.tokenizer.eos_token_id
            ):
                early_terminations_tokens.append(tokens)
                early_terminations_probs.append(log_probs)
            elif (
                len(best_beams_tokens) < k and tokens[-1] == self.tokenizer.eos_token_id
            ):
                best_beams_tokens.append(tokens)
                best_beams_tokens.append(log_probs)

        best_beams_tokens = t.tensor(best_beams_tokens)
        best_beams_probs = t.tensor(best_beams_probs)

        early_terminations_tokens = t.tensor(early_terminations_tokens)
        early_terminations_probs = t.tensor(early_terminations_probs)

        return (
            Beams(self.model, self.tokenizer, best_beams_tokens, best_beams_probs),
            Beams(
                self.model,
                self.tokenizer,
                early_terminations_tokens,
                early_terminations_probs,
            ),
        )

    # copied solution
    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: int | None,
        k: int,
    ) -> tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
        """
        logprobs:
            tensor of the log-probs for the next token
        no_repeat_ngram_size:
            size of ngram to avoid repeating
        k:
            number of top logits to return, for each beam in our collection

        Returns:
            equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure that no returned tokens would produce an
            ngram of size `no_repeat_ngram_size` which has already appeared in `self.tokens`.
        """
        batch, seq_len = self.tokens.shape

        # If completion isn't long enough for a repetition, or we have no restructions, just return topk
        if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size - 1):
            # Otherwise, we need to check for ngram repetitions
            # First, get the most recent `no_repeat_ngram_size-1` tokens
            last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size - 1) :]
            # Next, find all the tokens we're not allowed to generate, by checking all past ngrams for a match
            for i in range(seq_len - (no_repeat_ngram_size - 1)):
                ngrams = self.tokens[:, i : i + no_repeat_ngram_size]  # (batch, ngram)
                ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(
                    -1
                )  # (batch,)
                ngram_end_tokens = ngrams[:, [-1]]  # (batch, 1)
                # Fill logprobs with neginf wherever the ngrams are repeated
                logprobs[range(batch), ngram_end_tokens] = t.where(
                    ngrams_are_repeated,
                    -1.0e4,
                    logprobs[range(batch), ngram_end_tokens],
                )

        # Finally, get our actual tokens
        return logprobs.topk(k=k, dim=-1)

    def print(self, title="Best completions", max_print_chars=80) -> None:
        """
        Prints out a set of sequences with their corresponding logprob sums.
        """
        if len(self.tokens) == 0:
            return
        table = Table("logprob sum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = (
                    text[: int(0.3 * max_print_chars)]
                    + " ... "
                    + text[-int(0.7 * max_print_chars) :]
                )
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int | None = None,
) -> list[tuple[float, str]]:
    """
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
    prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
    we've generated `num_returns_sequences` terminating sequences.
    """
    assert num_return_sequences <= num_beams
    self.model.eval()

    tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

    final_logprobs_and_completions = (
        []
    )  # we add to this list as we get terminated beams
    best_beams = Beams(
        self.model, self.tokenizer, t.tensor([0.0]).to(device), tokens
    )  # start with just 1 beam

    for _ in tqdm(range(max_new_tokens)):
        t.cuda.empty_cache()

        # Generate & filter beams
        best_beams = best_beams.generate(
            k=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
        )
        best_beams, best_beams_terminated = best_beams.filter(k=num_beams)

        # Add terminated beams to our list, and return early if we have enough
        final_logprobs_and_completions.extend(
            best_beams_terminated.logprobs_and_completions
        )

        if len(final_logprobs_and_completions) >= num_return_sequences:
            return final_logprobs_and_completions[:num_return_sequences]

    # Return terminated beams plus the best ongoing beams of length `orig_len + max_new_tokens`
    final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
    return final_logprobs_and_completions[:num_return_sequences]


TransformerSampler.beam_search = beam_search

# Start with prompt "When I was", get top 3 tokens (and their logprobs), and use that to create & display the top 3 beams
prompt = "When I was"
tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
logprobs = model(tokens)[0, -1].log_softmax(-1)
top_logprobs, top_tokens = logprobs.topk(k=3, dim=-1)

new_tokens = t.concat([tokens.repeat(3, 1), top_tokens.unsqueeze(-1)], dim=-1)

beams = Beams(model, tokenizer, logprob_sums=top_logprobs, tokens=new_tokens)
beams.print()

print("Testing generate...")
new_beams = beams.generate(k=3, no_repeat_ngram_size=1)
new_beams.print()

expected_values = [
    (-3.1, "When I was a kid"),
    (-4.8, "When I was a child"),
    (-4.9, "When I was a little"),
]

for i, (logprob_sum, completion) in enumerate(new_beams.logprobs_and_completions[:3]):
    assert abs(logprob_sum - expected_values[i][0]) < 0.1, f"{i}"
    assert completion == expected_values[i][1], f"{i}"

print("All tests for `generate` passed!")

print("Testing `filter`...")

best_beams, terminated_beams = new_beams.filter(3)
best_beams.print()

expected_values = [
    (-3.1, "When I was a kid"),
    (-3.2, "When I was growing up"),
    (-4.6, "When I was in the"),
]

for i, (logprob_sum, completion) in enumerate(best_beams.logprobs_and_completions):
    assert abs(logprob_sum - expected_values[i][0]) < 0.1, f"{i}"
    assert completion == expected_values[i][1], f"{i}"

assert len(terminated_beams.logprobs_and_completions) == 0

print("All tests for `filter` passed!")

print("Testing `no_repeat_ngram_size`...")

new_beams = beams
for _ in range(5):
    new_beams = new_beams.generate(k=1)
new_beams.print(title="Completions with no ngram restriction")
assert all(
    "I was" in completion.removeprefix(prompt)
    for _, completion in new_beams.logprobs_and_completions
), "Without restriction, all beams should be completed as '...I was...'"

new_beams = beams
for _ in range(5):
    new_beams = new_beams.generate(k=1, no_repeat_ngram_size=2)
new_beams.print(title="Completions with no repeated bigrams")
assert all(
    "I was" not in completion.removeprefix(prompt)
    for _, completion in new_beams.logprobs_and_completions
), "With no repeated bigrams, no beams should contain a second '...I was...'"

# %%


sampler = TransformerSampler(model, tokenizer)

prompt = "Jingle bells, jingle bells, jingle all the way"
print(f"Testing greedy decoding\nPrompt:   {prompt!r}")

expected = (
    "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
)
output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)

print(f"Expected: {expected!r}\nActual:   {output!r}\n")
assert output == expected

print("Tests passed!")

# %%.
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097,
}
frequency_of_top_5 = defaultdict(int)

N = 10_000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits)
    frequency_of_top_5[tokenizer.decode(token)] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word]
    observed_freq = frequency_of_top_5[word] / N
    print(
        f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}"
    )
    assert (
        abs(observed_freq - expected_freq) < 0.01
    ), "Try increasing N if this fails by a small amount."

print("Tests passed!")
# %%
logits = t.tensor([1, 2]).log()

cold_logits = TransformerSampler.apply_temperature(logits, temperature=0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)

hot_logits = TransformerSampler.apply_temperature(logits, temperature=1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)

print("Tests passed!")

# %%
bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt")
logits = t.ones(tokenizer.vocab_size)
penalized_logits = TransformerSampler.apply_frequency_penalty(
    input_ids.squeeze(), logits, 2.0
)

assert (
    penalized_logits[5156].item() == -11
), "Expected 6 occurrences of ' baby' with leading space, 1-2*6=-11"
assert (
    penalized_logits[14801].item() == -5
), "Expected 3 occurrences of ' Baby' with leading space, 1-2*3=-5"

print("Tests passed!")
# %%
sampler = TransformerSampler(model, tokenizer)

N_RUNS = 10_000
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(frequency_penalty=100.0)),
    ("Negative freq penalty", dict(frequency_penalty=-3.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]

table = Table("Name", "Kwargs", "Output", title="Sampling - Manual Testing")

for name, kwargs in cases:
    for i in range(N_RUNS):
        output = sampler.sample(your_prompt, max_tokens_generated=24, **kwargs)
        table.add_row(name, str(kwargs), repr(output) + "\n")

rprint(table)
# %%
dist = t.distributions.categorical.Categorical(logits=t.tensor([1, 2, 3]))
dist.sample().item()
# %%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097,
}
topk_5_sum = sum(expected_top_5.values())

observed_freqs = defaultdict(int)

N = 10_000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_k=5)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word] / topk_5_sum
    observed_freq = observed_freqs[word] / N
    print(
        f"Word: {word!r:<9}. Expected freq = {expected_freq:.4f}, observed freq = {observed_freq:.4f}"
    )
    assert abs(observed_freq - expected_freq) < 0.01

# %%
sampler = TransformerSampler(model, tokenizer)

your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

output = sampler.sample(your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)

rprint(f"Your model said:\n\n[bold dark_orange]{output}")
# %%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_10pct = {
    " church": 0.0648,
    " house": 0.0367,  # These are the two most likely tokens, and add up to >10%
}
top_10pct_sum = sum(expected_top_10pct.values())

observed_freqs = defaultdict(int)

N = 5000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_p=0.1)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_10pct:
    expected_freq = expected_top_10pct[word] / top_10pct_sum
    observed_freq = observed_freqs[word] / N
    print(
        f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}"
    )
    assert (
        abs(observed_freq - expected_freq) < 0.01
    ), "Try increasing N if this fails by a small amount."


# %%
@dataclass
class SampleClass:
    a: int
    b: float


s = SampleClass(a=2, b=3)
s
# %%
