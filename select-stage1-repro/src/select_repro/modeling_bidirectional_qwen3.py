from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.loss.loss_utils import ForMaskedLMLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    AttentionMaskConverter,
    Cache,
    DynamicCache,
    FlashAttentionKwargs,
    KwargsForCausalLM,
    Qwen3Attention,
    Qwen3Model,
    Qwen3PreTrainedModel,
    StaticCache,
    SlidingWindowCache,
    apply_rotary_pos_emb,
    can_return_tuple,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack


class Qwen3BidirectionalAttention(Qwen3Attention):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                attention_interface = eager_attention_forward
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attention_kwargs = dict(kwargs)
        if self.config._attn_implementation == "sdpa":
            attention_kwargs["is_causal"] = False

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **attention_kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3BidirectionalModel(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn = Qwen3BidirectionalAttention(config=config, layer_idx=layer_idx)

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if attention_mask is None:
            return None

        if attention_mask.dim() == 4:
            return attention_mask.to(dtype=input_tensor.dtype)

        if attention_mask.dim() != 2:
            raise ValueError(f"Unsupported attention mask shape for bidirectional Qwen3: {attention_mask.shape}")

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        sequence_length = input_tensor.shape[1]
        if using_static_cache or using_sliding_window_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else past_seen_tokens + sequence_length

        if attention_mask.shape[-1] != target_length:
            raise ValueError(
                "Bidirectional Qwen3 expects attention_mask to cover the full key/value length when caching is used."
            )

        mask = attention_mask[:, None, None, :].to(device=input_tensor.device)
        mask = mask.expand(-1, 1, sequence_length, -1)
        valid = mask.to(dtype=torch.bool) if mask.dtype == torch.bool else mask > 0
        bidirectional_mask = torch.zeros(mask.shape, dtype=input_tensor.dtype, device=input_tensor.device)
        bidirectional_mask.masked_fill_(~valid, torch.finfo(input_tensor.dtype).min)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            bidirectional_mask = AttentionMaskConverter._unmask_unattended(
                bidirectional_mask, torch.finfo(input_tensor.dtype).min
            )

        return bidirectional_mask


class Qwen3ForBidirectionalMaskedLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    loss_type = "ForMaskedLM"

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3BidirectionalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_full_logits: Optional[bool] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_full_logits = labels is None if return_full_logits is None else return_full_logits

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = None
        loss = None
        masked_token_count = None

        if labels is None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            masked_positions = labels != -100
            masked_token_count = int(masked_positions.sum().item())

            if return_full_logits:
                logits = self.lm_head(hidden_states)
                loss = ForMaskedLMLoss(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            else:
                selected_hidden_states = hidden_states[masked_positions]
                logits = self.lm_head(selected_hidden_states)
                if masked_token_count == 0:
                    loss = hidden_states.new_zeros(())
                else:
                    selected_labels = labels[masked_positions]
                    loss = ForMaskedLMLoss(
                        logits=logits,
                        labels=selected_labels,
                        vocab_size=self.config.vocab_size,
                        **kwargs,
                    )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output.logits_are_full_sequence = return_full_logits or labels is None
        output.masked_token_count = masked_token_count
        return output
