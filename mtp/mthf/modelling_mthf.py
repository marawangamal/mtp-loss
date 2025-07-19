import copy
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from mtp.mheads import MHEADS
from mtp._types import ModelHeadType
from mtp.mheads._abc import AbstractDisributionHeadConfig


class MultiTokenHFConfig(PretrainedConfig):
    model_type = "mthf"

    def __init__(
        self,
        model_name: str = "gpt2",
        horizon: int = 1,
        rank: int = 1,
        model_head: ModelHeadType = "stp",
        pretrained: bool = False,
        lambda_mhead: float = 1.0,  # weight for mhead loss
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.horizon = horizon
        self.model_head = model_head
        self.rank = rank
        self.pretrained = pretrained
        self.lambda_mhead = lambda_mhead


class MultiTokenHF(PreTrainedModel, GenerationMixin):
    config_class = MultiTokenHFConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MultiTokenHFConfig):
        super().__init__(config)

        # Set dims
        self.vocab_size, self.embedding_dim = get_model_dims(config.model_name)
        self.horizon = config.horizon

        # Set backbone
        self.backbone, self.lm_head = get_backbone(config.model_name, config.pretrained)

        # Set multi-token head
        mhead_config = AbstractDisributionHeadConfig(
            d_model=self.embedding_dim,
            d_output=self.vocab_size,
        )
        self.mhead = MHEADS[config.model_head](mhead_config)

    def get_output_embeddings(self):
        return self.lm_head.weight

    def get_input_embeddings(self):
        # Try to get input embeddings from the backbone
        if hasattr(self.backbone, "wte"):
            return self.backbone.wte
        elif hasattr(self.backbone, "embed_tokens"):
            return self.backbone.embed_tokens
        else:
            raise NotImplementedError("Input embeddings not found in backbone.")

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.weight = new_embeddings

    def set_input_embeddings(self, new_embeddings):
        if hasattr(self.backbone, "wte"):
            self.backbone.wte = new_embeddings
        elif hasattr(self.backbone, "embed_tokens"):
            self.backbone.embed_tokens = new_embeddings
        else:
            raise NotImplementedError("Input embeddings not found in backbone.")

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Standard for decoder-only models: just return input_ids and any attention_mask
        model_inputs = {"input_ids": input_ids}
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            model_inputs["attention_mask"] = kwargs["attention_mask"]
        return model_inputs

    def adjust_logits_during_generation(self, logits, **kwargs):
        # No adjustment by default
        return logits

    def forward(self, input_ids, labels=None, use_memory_efficient_loss=True, **kwargs):
        # Get hidden states from model
        outputs = self.backbone(input_ids=input_ids, **kwargs)
        z = outputs.last_hidden_state  # Hidden states. Shape: (B, T, D)

        # Compute loss if labels provided
        if labels is not None:
            T, V = z.shape[1], self.vocab_size
            if T <= self.horizon:
                raise ValueError(
                    f"Input sequence length ({T}) must be greater than horizon ({self.horizon}) for loss computation."
                )
            # Remove last position since we can't predict it
            z_lm = z[:, :-1]  # (B, T-1, D)
            y_lm = input_ids[:, 1:]  # (B, T-1)
            logits = self.lm_head(z_lm)
            loss_lm = torch.nn.functional.cross_entropy(
                logits.reshape(-1, V),
                y_lm.reshape(-1),
                reduction="mean",
            )
            return CausalLMOutput(loss=loss_lm, logits=logits)

        # For inference: return logits from last position
        logits = self.lm_head(z[:, -1:, :])  # (B, 1, D)
        return CausalLMOutput(logits=logits)


def get_model_dims(model_name: str) -> tuple[int, int]:
    """Get vocabulary size and embedding dimension from model config."""
    hf_config = AutoConfig.from_pretrained(model_name)
    vocab_size = hf_config.vocab_size
    embedding_dim = [  # ad-hoc to support both GPT and Llama
        getattr(hf_config, k)
        for k in ["n_embd", "hidden_size"]
        if hasattr(hf_config, k)
    ][0]
    return vocab_size, embedding_dim


def get_backbone(
    model_name: str, pretrained: bool = False
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Get a randomly initialized backbone of a HuggingFace model."""
    if pretrained:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        hf_model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name)
        )

    if hasattr(hf_model, "transformer"):
        return hf_model.transformer, hf_model.lm_head
    elif hasattr(hf_model, "model"):  # e.g., Llama
        return hf_model.model, hf_model.lm_head
    else:
        raise ValueError(f"Cannot find transformer/model backbone in {type(hf_model)}")
