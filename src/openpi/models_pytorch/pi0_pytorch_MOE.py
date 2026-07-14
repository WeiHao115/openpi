import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class ForceEncoder(nn.Module):
    def __init__(self, window_size: int, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(window_size * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.reshape(x.shape[0], -1))


class TactileEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.reshape(x.shape[0], -1)).unsqueeze(1)


class TokenMoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2, hidden_mult: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(dim, num_experts)
        hidden_dim = dim * hidden_mult
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route_logits = self.router(x)
        top_values, top_indices = torch.topk(route_logits, self.top_k, dim=-1)
        route_weights = torch.zeros_like(route_logits)
        route_weights.scatter_(-1, top_indices, F.softmax(top_values, dim=-1))
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return torch.sum(expert_out * route_weights.unsqueeze(-1), dim=-2)


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        self.force_dim = getattr(config, "force_dim", 6)
        self.force_window_size = getattr(config, "force_window_size", 1)
        self.tactile_dim = getattr(config, "tactile_dim", 1)
        self.force_encoder = ForceEncoder(self.force_window_size, self.force_dim, 256, action_expert_config.width)
        self.tactile_encoder = TactileEncoder(self.tactile_dim, 32, paligemma_config.width)
        self.image_input_proj = nn.Identity() if paligemma_config.width == 2048 else nn.Linear(2048, paligemma_config.width)
        self.force_in_proj = nn.Linear(self.force_dim, action_expert_config.width)
        self.force_out_proj = nn.Linear(action_expert_config.width, self.force_dim)

        moe_num_experts = getattr(config, "moe_num_experts", 4)
        moe_top_k = getattr(config, "moe_top_k", 2)
        moe_hidden_mult = getattr(config, "moe_hidden_mult", 2)
        self.image_moe = TokenMoE(paligemma_config.width, moe_num_experts, moe_top_k, moe_hidden_mult)
        self.text_moe = TokenMoE(paligemma_config.width, moe_num_experts, moe_top_k, moe_hidden_mult)
        self.tactile_moe = TokenMoE(paligemma_config.width, moe_num_experts, moe_top_k, moe_hidden_mult)
        self.force_moe = TokenMoE(action_expert_config.width, moe_num_experts, moe_top_k, moe_hidden_mult)
        self.action_moe = TokenMoE(action_expert_config.width, moe_num_experts, moe_top_k, moe_hidden_mult)
        self.force_to_image = nn.Linear(action_expert_config.width, paligemma_config.width)
        self.image_to_force = nn.Linear(paligemma_config.width, action_expert_config.width)
        self.image_force_fuse = nn.Linear(paligemma_config.width * 2, paligemma_config.width)
        self.force_image_fuse = nn.Linear(action_expert_config.width * 2, action_expert_config.width)
        self.last_predicted_forces = None

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation,
            train=train,
            image_keys=list(observation.images.keys()),
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def _batch_size_device(self, observation):
        return observation.state.shape[0], observation.state.device

    def _get_optional_tensor(self, observation, names):
        for name in names:
            value = getattr(observation, name, None)
            if value is not None:
                return value
        return None

    def _preprocess_force(self, observation):
        bsize, device = self._batch_size_device(observation)
        force = self._get_optional_tensor(observation, ("force", "forces", "force_torque"))
        if force is None:
            force = torch.zeros(bsize, self.force_window_size, self.force_dim, dtype=torch.float32, device=device)
        else:
            force = force.to(device=device, dtype=torch.float32)
            if force.ndim == 2:
                force = force[:, None, :]
            if force.shape[-1] < self.force_dim:
                pad = torch.zeros(*force.shape[:-1], self.force_dim - force.shape[-1], device=device, dtype=force.dtype)
                force = torch.cat([force, pad], dim=-1)
            force = force[..., : self.force_dim]
            if force.shape[1] < self.force_window_size:
                pad_len = self.force_window_size - force.shape[1]
                pad = force[:, :1].expand(bsize, pad_len, self.force_dim)
                force = torch.cat([pad, force], dim=1)
            force = force[:, -self.force_window_size :, :]

        mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        return [force], [mask]

    def _preprocess_tactile(self, observation):
        bsize, device = self._batch_size_device(observation)
        tactile = self._get_optional_tensor(observation, ("tactile", "tactile_signal", "tactile_signals"))
        if tactile is None:
            tactile = torch.zeros(bsize, self.tactile_dim, dtype=torch.float32, device=device)
        else:
            tactile = tactile.to(device=device, dtype=torch.float32).reshape(bsize, -1)
            if tactile.shape[-1] < self.tactile_dim:
                pad = torch.zeros(bsize, self.tactile_dim - tactile.shape[-1], device=device, dtype=tactile.dtype)
                tactile = torch.cat([tactile, pad], dim=-1)
            tactile = tactile[:, : self.tactile_dim]

        mask = torch.ones(bsize, dtype=torch.bool, device=device)
        return [tactile], [mask]

    def encode_force_context(self, forces):
        force_embs = []
        for force_seq in forces:
            def force_embed_func(f_seq):
                return self.force_encoder(f_seq).unsqueeze(1)

            force_embs.append(self._apply_checkpoint(force_embed_func, force_seq))
        return torch.cat(force_embs, dim=1)

    def force_target_from_observation(self, observation, forces, force_targets=None):
        if force_targets is None:
            force_targets = self._get_optional_tensor(observation, ("force_target", "target_force", "force_targets"))
        if force_targets is None:
            return torch.stack([force_seq[:, -1, :] for force_seq in forces], dim=1)

        force_targets = force_targets.to(device=forces[0].device, dtype=torch.float32)
        if force_targets.ndim == 2:
            force_targets = force_targets[:, None, :]
        if force_targets.shape[-1] < self.force_dim:
            pad = torch.zeros(
                *force_targets.shape[:-1],
                self.force_dim - force_targets.shape[-1],
                device=force_targets.device,
                dtype=force_targets.dtype,
            )
            force_targets = torch.cat([force_targets, pad], dim=-1)
        return force_targets[..., : self.force_dim]

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images,
        img_masks,
        tactile_signals,
        tactile_masks,
        forces,
        force_masks,
        lang_tokens,
        lang_masks,
        force_context=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed image/tactile/force/language tokens before the VLM/Gemma backbone."""
        embs = []
        pad_masks = []
        att_masks = []
        image_chunks = []

        if force_context is None:
            force_context = self.encode_force_context(forces)
        force_summary = self.force_to_image(force_context.mean(dim=1))

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            if img.ndim == 4 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2).contiguous()

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            img_emb = img_emb.to(dtype=self.image_force_fuse.weight.dtype)
            img_emb = self.image_input_proj(img_emb)

            bsize, num_img_embs = img_emb.shape[:2]
            force_for_image = force_summary[:, None, :].expand(bsize, num_img_embs, -1)
            img_emb = self.image_force_fuse(torch.cat([img_emb, force_for_image], dim=-1))
            img_emb = self.image_moe(img_emb)
            image_chunks.append(img_emb)

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        tactile_chunks = []
        for tactile_signal, tactile_mask in zip(tactile_signals, tactile_masks, strict=True):

            def tactile_embed_func(tactile_signal):
                return self.tactile_encoder(tactile_signal)

            tactile_emb = self._apply_checkpoint(tactile_embed_func, tactile_signal)
            tactile_emb = self.tactile_moe(tactile_emb)
            bsize, num_tactile_embs = tactile_emb.shape[:2]
            tactile_chunks.append(tactile_emb)

            embs.append(tactile_emb)
            pad_masks.append(tactile_mask[:, None].expand(bsize, num_tactile_embs))
            att_masks += [0] * num_tactile_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        lang_emb = lang_emb.to(dtype=self.text_moe.router.weight.dtype)
        lang_emb = self.text_moe(lang_emb)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        summary_chunks = image_chunks + tactile_chunks
        image_summary = torch.cat(summary_chunks, dim=1).mean(dim=1) if summary_chunks else None
        return embs, pad_masks, att_masks, image_summary

    def embed_suffix(self, state, noisy_actions, timestep, force_context=None, noisy_forces=None, image_summary=None):
        """Embed state/force/action tokens before the Gemma expert."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        if self.pi05:
            if force_context is None or noisy_forces is None:
                raise ValueError("pi05 MOE requires force_context and noisy_forces in embed_suffix.")
            if force_context.shape[1] != noisy_forces.shape[1]:
                force_context = force_context.mean(dim=1, keepdim=True).expand(
                    noisy_forces.shape[0], noisy_forces.shape[1], -1
                )
            force_emb = self.force_in_proj(noisy_forces) + force_context
            if image_summary is not None:
                image_for_force = self.image_to_force(image_summary)[:, None, :].expand_as(force_emb)
                force_emb = self.force_image_fuse(torch.cat([force_emb, image_for_force], dim=-1))
            force_emb = self.force_moe(force_emb)
            embs.append(force_emb)
            bsize, num_force_tokens = force_emb.shape[:2]
            force_mask = torch.ones(bsize, num_force_tokens, dtype=torch.bool, device=timestep.device)
            pad_masks.append(force_mask)
            att_masks += [1] + ([0] * (num_force_tokens - 1))

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        action_emb = self.action_moe(action_emb)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None, force_targets=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        tactile_signals, tactile_masks = self._preprocess_tactile(observation)
        forces, force_masks = self._preprocess_force(observation)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        force_context = self.encode_force_context(forces)
        if self.pi05:
            force_targets = self.force_target_from_observation(observation, forces, force_targets)
            force_targets = force_targets.to(device=actions.device, dtype=torch.float32)
            force_noise = self.sample_noise(force_targets.shape, actions.device)
            force_x_t = time_expanded * force_noise + (1 - time_expanded) * force_targets
            force_u_t = force_noise - force_targets
        else:
            force_targets = None
            force_x_t = None
            force_u_t = None

        prefix_embs, prefix_pad_masks, prefix_att_masks, image_summary = self.embed_prefix(
            images,
            img_masks,
            tactile_signals,
            tactile_masks,
            forces,
            force_masks,
            lang_tokens,
            lang_masks,
            force_context,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            time,
            force_context,
            force_x_t,
            image_summary,
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out.to(dtype=torch.float32)
        if self.pi05:
            force_out = suffix_out[:, : force_targets.shape[1]]
            action_out = suffix_out[:, -self.config.action_horizon :]
        else:
            action_out = suffix_out[:, -self.config.action_horizon :]

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(action_out):
            return self.action_out_proj(action_out)

        v_t = self._apply_checkpoint(action_out_proj_func, action_out)

        action_loss = F.mse_loss(u_t, v_t, reduction="none")
        if self.pi05:
            force_v_t = self.force_out_proj(force_out)
            force_loss = F.mse_loss(force_u_t, force_v_t, reduction="none").mean(dim=(1, 2))
            return action_loss + force_loss[:, None, None]
        return action_loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        tactile_signals, tactile_masks = self._preprocess_tactile(observation)
        forces, force_masks = self._preprocess_force(observation)
        force_context = self.encode_force_context(forces)
        force_x_t = self.sample_noise((bsize, self.config.action_horizon, self.force_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, image_summary = self.embed_prefix(
            images,
            img_masks,
            tactile_signals,
            tactile_masks,
            forces,
            force_masks,
            lang_tokens,
            lang_masks,
            force_context,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            denoise_out = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                force_context,
                force_x_t,
                image_summary,
            )
            if self.pi05:
                force_v_t, v_t = denoise_out
                force_x_t = force_x_t + dt * force_v_t
            else:
                v_t = denoise_out

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        self.last_predicted_forces = force_x_t if self.pi05 else None
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        force_context=None,
        force_x_t=None,
        image_summary=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            timestep,
            force_context,
            force_x_t,
            image_summary,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out.to(dtype=torch.float32)
        if self.pi05:
            force_out = suffix_out[:, : force_x_t.shape[1]]
            action_out = suffix_out[:, -self.config.action_horizon :]
            return self.force_out_proj(force_out), self.action_out_proj(action_out)

        action_out = suffix_out[:, -self.config.action_horizon :]
        return self.action_out_proj(action_out)
