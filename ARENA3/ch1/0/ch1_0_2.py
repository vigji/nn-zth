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

dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")

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
    dataset_dict["train"], batch_size=cfg_trainer.batch_size, shuffle=True, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    dataset_dict["test"], batch_size=cfg_trainer.batch_size, shuffle=False, num_workers=0, pin_memory=True
)

## for predictions:
def sampling_fn(model: DemoTransformer, prompt: str) -> str:
    sampler = solutions.TransformerSampler(model, reference_gpt2.tokenizer)
    output = sampler.sample(prompt, temperature=0.7, top_p=0.95, max_tokens_generated=16)
    return output

class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, 
                 model: DemoTransformer,
                 prompt_list = None):
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

        self.prompt_list = prompt_list

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
            wandb.init(project=self.args.wandb_project, 
                       name=self.args.wandb_name, 
                       config=full_config)
        
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        inferences_list = []
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):

                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()

            if self.prompt_list is not None:
                sampled = [sampling_fn(model, prompt=prompt) for prompt in self.prompt_list]
                
                inferences_list.append([self.step, epoch, ] + sampled)
                inf_table = wandb.Table(columns=["step", "epoch", ] + [f"text{i}" for i in range(len(sampled))],
                                        data=inferences_list)
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
        token_ids = t.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt)))
        
        for _ in range(max_tokens_generated):
            logits = model(token_ids.unsqueeze(0))
            next_token = self.sample_next_token(token_ids, logits, **kwargs)


            if next_token == self.tokenizer.eos_token_id:
                break

            token_ids = t.concatenate([token_ids, t.tensor([next_token])])
            
            output = self.tokenizer.convert_ids_to_tokens(token_ids)
            output = "".join(output).replace("Ġ", " ")
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
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

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
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
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
        return logits[0, -1, :].argmax().item()

    @staticmethod
    def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        raise NotImplementedError()

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        raise NotImplementedError()

    @staticmethod
    def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()


sampler = TransformerSampler(model, tokenizer)

prompt = "Jingle bells, jingle bells, jingle all the way"
print(f"Testing greedy decoding\nPrompt:   {prompt!r}")

expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)

print(f"Expected: {expected!r}\nActual:   {output!r}\n")
assert output == expected

print("Tests passed!")
# %%.
tokenizer.tokenize("A noisy cat")
# %%
?tokenizer.tokenize
# %%
token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("A noisy cat"))
token_ids = t.tensor(token_ids).unsqueeze(0)
logits = model(token_ids)
greedy_max = logits[:, -1, :].argmax(3)
# t.concatenate([token_ids, t.tensor([[3]])], axis=1)
# %%
token_ids.shape
# %%
t.tensor([[3]]).shape
# %%
