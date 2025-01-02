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
        if self.cfg.debug:
            print("residual shape: ", residual.shape)

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
        if self.cfg.debug:
            print("Input shape: ", tokens.shape)
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
        if self.cfg.debug:
            print("Input shape: ", tokens.shape)

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

    def forward(self, resid_pre: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
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
        
        return einops.einsum(normalized_resid_final, self.W_U, 
                      "batch position d_model, d_model d_vocab -> batch position d_vocab") + self.b_U



class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        embedded = self.embed(tokens)
        pos_embedded = self.pos_embed(tokens)

        transformer_input = embedded + pos_embedded
        for trasf_block in self.blocks:
            transformer_input = trasf_block(transformer_input)

        return self.unembed(self.ln_final(transformer_input))

# %%
########################
# Training a Tranformer!
########################


cfg = Config()

model_cfg = Config(
    debug=False,
    d_model=256,
    n_heads=4,
    d_head=64,
    d_mlp=1024,
    n_layers=2,
    n_ctx=256,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg)

@dataclass
class TransformerTrainingArgs:
    batch_size = 16
    epochs = 20
    max_steps_per_epoch = 100
    lr = 1e-3
    weight_decay = 1e-2
    use_wandb = True
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None


args = TransformerTrainingArgs()

# %%
dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
print(dataset)
print(dataset[0]["text"][:100])
# %%
tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=4,
)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(
    dataset_dict["train"], batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    dataset_dict["test"], batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
)

# %%
first_batch = train_loader.dataset[: args.batch_size]

print(first_batch.keys())
print(first_batch["tokens"].shape)
# %%
class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args

        self.optimizer = t.optim.AdamW(self.model.parameters(), 
                                       lr=args.lr, 
                                       weight_decay=args.weight_decay)
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

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        self.optimizer.zero_grad()

        tokens = batch["tokens"].to(device)
        demo_logits = self.model(tokens)
        log_softmax = demo_logits.log_softmax(dim=-1)

        gathered_probs = log_softmax.gather(2, 
                                            tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

        loss = -gathered_probs.mean()

        loss.backward()
        self.optimizer.step()

        wandb.log(dict(loss=loss), step=self.step)
        self.step += 1

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

        wandb.log(dict(loss=accuracy), step=self.step)

        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):

                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()

        if self.args.use_wandb:
            wandb.finish()

model = DemoTransformer(model_cfg).to(device)
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model)
trainer.train()
# %%
test_loader = DataLoader(
    dataset_dict["test"], batch_size=args.batch_size, 
    shuffle=False, num_workers=0, pin_memory=True
)

model = DemoTransformer(model_cfg).to(device)

for i, batch in enumerate(test_loader):
    logits = model(batch["tokens"])
    log_softmax = logits.log_softmax(dim=-1)
    if i == 0:
        break
# See the full run here: https://api.wandb.ai/links/callum-mcdougall/4xtin05h

# %%
prediction = logits.argmax(2)
prediction[:, :-1].shape
# %%
batch["tokens"].shape

# %%
(prediction[:, :-1] == batch["tokens"][:, 1:].to(device)).sum() / prediction[:, :-1].numel()
# %%
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model)
trainer.train()