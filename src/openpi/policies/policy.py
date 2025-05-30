from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias
from PIL import Image

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import image_tools
from openpi.shared import nnx_utils

from ..models.model import Observation
from ..models.contrast_utils.contrast_image_generator import ContrastImageGenerator

BasePolicy: TypeAlias = _base_policy.BasePolicy


IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb"
)
IMAGE_RESOLUTION = (224, 224)

CONTRAST_IMAGE_GENERATOR_CONFIGS = {
    "base_0_rgb": {
        "by": "grounded_sam_tracking",
        "inpaint_mode": "lama",
    },
}
_CATCH_THRESHOLD = 69000 * 0.7

def preprocess_contrast_observation(
    contrast_image_generators,
    observation,
    images,
    prompt,
    *,
    image_keys=IMAGE_KEYS,
    image_resolution=IMAGE_RESOLUTION,
):
    if not set(image_keys).issubset(images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(images)}")

    batch_shape = observation.state.shape[:-1]

    out_images, out_images_contrast = {}, {}
    for key in image_keys:
        image = images[key]

        if key not in contrast_image_generators:
            print(f"Key {key} not in contrast_image_generators, using original image.")
            contrast_image = image
        else:
            contrast_image = contrast_image_generators[key].generate(image[..., ::-1], prompt)[..., ::-1]
        
        if key == 'base_0_rgb':
            Image.fromarray(image).save('test.jpg')
            Image.fromarray(contrast_image).save('test_contrast.jpg')
        
        image = jnp.asarray(image[np.newaxis, ...])
        contrast_image = jnp.asarray(contrast_image[np.newaxis, ...])
        
        if image.shape[1:3] != image_resolution:
            image = image_tools.resize_with_pad(image, *image_resolution)
        
        if contrast_image.shape[1:3] != image_resolution:
            contrast_image = image_tools.resize_with_pad(contrast_image, *image_resolution)


        image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
        contrast_image = contrast_image.astype(np.float32) / 255.0 * 2.0 - 1.0
        
        out_images[key] = image
        out_images_contrast[key] = contrast_image
    
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    obs = Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
    contrast_obs = Observation(
        images=out_images_contrast,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
    return obs, contrast_obs


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.model = model
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        obs = Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, obs, **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class ContrastPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrast_image_generators = {
            key: ContrastImageGenerator(**CONTRAST_IMAGE_GENERATOR_CONFIGS[key])
            for key in CONTRAST_IMAGE_GENERATOR_CONFIGS
        }

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        prompt = obs['prompt']
        timestep = obs.pop('timestep')
        
        if timestep == 0:
            # Reset the contrast image generators at the start of each episode.
            print("Resetting contrast image generators.")
            for generator in self.contrast_image_generators.values():
                generator.reset()
        
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        images = inputs["image"]

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        obs = Observation.from_dict(inputs)
        base_obs, contrast_obs = preprocess_contrast_observation(
            self.contrast_image_generators,
            obs,
            images,
            prompt,
        )
        
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, base_obs, contrast_obs, **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
