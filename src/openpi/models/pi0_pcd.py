import einops
import jax
import jax.numpy as jnp
from flax import nnx
from typing_extensions import override

from .model import Observation
from .pi0 import Pi0, Pi0Config, make_attn_mask
from .contrast_utils.kde_contrast_decoding import ContrastDecoding


NUM_REPEATS = 12
ALPHA = 0.2
BANDWIDTH_FACTOR = 1.0
KEEP_THRESHOLD = 0.5


def repeat_observation(obs, num_repeats):
    images = {k: jnp.repeat(v, num_repeats, axis=0) for k, v in obs.images.items()}
    image_masks = {k: jnp.repeat(v, num_repeats, axis=0) for k, v in obs.image_masks.items()}
    state = jnp.repeat(obs.state, num_repeats, axis=0)
    tokenized_prompt = jnp.repeat(obs.tokenized_prompt, num_repeats, axis=0)
    tokenized_prompt_mask = jnp.repeat(obs.tokenized_prompt_mask, num_repeats, axis=0)
    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


class Pi0PCDConfig(Pi0Config):
    @override
    def create(self, rng):
        return Pi0PCD(self, rngs=nnx.Rngs(rng))


class Pi0PCD(Pi0):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_repeats = NUM_REPEATS
        self.contrast_decoding = ContrastDecoding(alpha=ALPHA,
                                                  bandwidth_factor=BANDWIDTH_FACTOR,
                                                  keep_threshold=KEEP_THRESHOLD,
                                                  mode='jax')
    
    @override
    def sample_actions(
        self,
        rng,
        observation,
        contrast_observation,
        *,
        num_steps=10,
    ):
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        assert batch_size == 1
        
        noise = jax.random.normal(rng, (self.num_repeats, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        contrast_prefix_tokens, contrast_prefix_mask, contrast_prefix_ar_mask = self.embed_prefix(contrast_observation)
        contrast_prefix_attn_mask = make_attn_mask(contrast_prefix_mask, contrast_prefix_ar_mask)
        contrast_positions = jnp.cumsum(contrast_prefix_mask, axis=1) - 1
        _, contrast_kv_cache = self.PaliGemma.llm([contrast_prefix_tokens, None], mask=contrast_prefix_attn_mask, positions=contrast_positions)

        prefix_tokens = jnp.repeat(prefix_tokens, self.num_repeats, axis=0)
        prefix_mask = jnp.repeat(prefix_mask, self.num_repeats, axis=0)
        prefix_ar_mask = jnp.repeat(prefix_ar_mask, self.num_repeats, axis=0)
        kv_cache = (jnp.repeat(kv_cache[0], self.num_repeats, axis=1), jnp.repeat(kv_cache[1], self.num_repeats, axis=1))

        contrast_prefix_tokens = jnp.repeat(contrast_prefix_tokens, self.num_repeats, axis=0)
        contrast_prefix_mask = jnp.repeat(contrast_prefix_mask, self.num_repeats, axis=0)
        contrast_prefix_ar_mask = jnp.repeat(contrast_prefix_ar_mask, self.num_repeats, axis=0)
        contrast_kv_cache = (jnp.repeat(contrast_kv_cache[0], self.num_repeats, axis=1), jnp.repeat(contrast_kv_cache[1], self.num_repeats, axis=1))

        observation = repeat_observation(observation, self.num_repeats)
        contrast_observation = repeat_observation(contrast_observation, self.num_repeats)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, self.num_repeats))
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            assert full_attn_mask.shape == (
                self.num_repeats,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache)

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt
        
        def contrast_step(carry):
            x_t, time = carry
            contrast_suffix_tokens, contrast_suffix_mask, contrast_suffix_ar_mask = self.embed_suffix(contrast_observation, x_t, jnp.broadcast_to(time, self.num_repeats))

            contrast_suffix_attn_mask = make_attn_mask(contrast_suffix_mask, contrast_suffix_ar_mask)
            contrast_prefix_attn_mask = einops.repeat(contrast_prefix_mask, "b p -> b s p", s=contrast_suffix_tokens.shape[1])
            contrast_full_attn_mask = jnp.concatenate([contrast_prefix_attn_mask, contrast_suffix_attn_mask], axis=-1)

            assert contrast_full_attn_mask.shape == (
                self.num_repeats,
                contrast_suffix_tokens.shape[1],
                contrast_prefix_tokens.shape[1] + contrast_suffix_tokens.shape[1],
            )
            contrast_positions = jnp.sum(contrast_prefix_mask, axis=-1)[:, None] + jnp.cumsum(contrast_suffix_mask, axis=-1) - 1
            (contrast_prefix_out, contrast_suffix_out), _ = self.PaliGemma.llm([None, contrast_suffix_tokens], mask=contrast_full_attn_mask, positions=contrast_positions, kv_cache=contrast_kv_cache)

            assert contrast_prefix_out is None
            contrast_v_t = self.action_out_proj(contrast_suffix_out[:, -self.action_horizon :])
            return x_t + dt * contrast_v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        contrast_actions, _ = jax.lax.while_loop(cond, contrast_step, (noise, 1.0))
        actions = self.contrast_decoding(actions, contrast_actions)
        return actions
