from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image, ImageOps

from .config import EnvisionConfig
from .ddim_inverter import DDIMInverter
from .ddim_reconstructor import DDIMReconstructor
from .gradient_estimator import TweedieGradientEstimator
from .langevin import LangevinPerturber
from .latent_codec import LatentCodec
from .model_loader import StableDiffusionLoader
from .output_writer import EnvisionOutputWriter
from .preprocessor import ImagePreprocessor
from .prompt_conditioner import PromptConditioner
from .representative import RepresentativeSelector
from .uncertainty import UncertaintyEstimator


@dataclass
class EnvisionResult:
    original_image_path: str
    impression_image_path: str
    uncertainty_map_path: str
    uncertainty_heatmap_path: str
    metadata_path: str


class EnvisionPipeline:
    def __init__(self, config: EnvisionConfig) -> None:
        self.config = config

    def run(self) -> EnvisionResult:
        self.config.validate()
        started = time.time()

        preprocessor = ImagePreprocessor(self.config.image_size)
        prep = preprocessor.run(self.config.input_image)
        original_image = ImageOps.exif_transpose(Image.open(self.config.input_image)).convert("RGB")

        components = StableDiffusionLoader(
            self.config.sd_model_dir,
            dtype=self.config.dtype,
            device=self.config.device,
        ).load()
        conditioner = PromptConditioner(
            components.tokenizer,
            components.text_encoder,
            components.device,
            components.dtype,
        )
        embeddings = conditioner.encode(self.config.prompt, self.config.negative_prompt)

        codec = LatentCodec(components.vae, components.device, components.dtype)
        z0 = codec.encode(prep.image_tensor)

        inverter = DDIMInverter(components.unet, components.scheduler, self.config.guidance_scale)
        inversion = inverter.invert(
            z0,
            embeddings.text_embeddings,
            self.config.num_ddim_steps,
            self.config.inversion_step_T,
            embeddings.negative_text_embeddings,
        )

        reconstructor = DDIMReconstructor(
            components.unet,
            components.scheduler,
            codec,
            self.config.guidance_scale,
        )
        no_perturb = reconstructor.reconstruct(
            inversion.zT,
            inversion.step_index_T,
            embeddings.text_embeddings,
            embeddings.negative_text_embeddings,
        )

        gradient = TweedieGradientEstimator(components.unet, components.scheduler, self.config.guidance_scale)
        perturber = LangevinPerturber(gradient)
        sample_images = []
        sample_debug = []
        for sample_idx in range(self.config.sample_count_K):
            perturbed = perturber.perturb(
                inversion.zT,
                inversion.timestep_T,
                embeddings.text_embeddings,
                embeddings.negative_text_embeddings,
                self.config.langevin_steps_M,
                self.config.eta_start,
                self.config.eta_end,
                self.config.temperature_tau,
                self.config.seed + sample_idx,
            )
            reconstructed = reconstructor.reconstruct(
                perturbed.latent,
                inversion.step_index_T,
                embeddings.text_embeddings,
                embeddings.negative_text_embeddings,
            )
            sample_images.append(reconstructed.image)
            sample_debug.append({"sample_index": sample_idx, "langevin_steps": perturbed.debug_steps})

        uncertainty = UncertaintyEstimator().estimate(sample_images)
        selector = RepresentativeSelector()
        representative = selector.select(prep.image_pil, sample_images)
        difference = selector.difference_image(prep.image_pil, representative.image)

        writer = EnvisionOutputWriter(self.config.output_dir)
        image_paths = writer.save_images(
            original_image,
            prep.image_pil,
            sample_images,
            representative.image,
            difference,
            uncertainty.gray_image,
            uncertainty.heatmap_image,
            uncertainty.uncertainty_map,
        )
        no_perturb_path = writer.output_dir / "reconstruction_no_perturb.png"
        no_perturb.image.save(no_perturb_path)

        metadata = {
            "transform_meta": prep.transform_meta,
            "device": str(components.device),
            "runtime_dtype": str(components.dtype),
            "timestep_T": inversion.timestep_T,
            "step_index_T": inversion.step_index_T,
            "representative_index": representative.index,
            "sample_diff_scores": representative.diff_scores,
            "sample_debug": sample_debug,
            "reconstruction_no_perturb": str(no_perturb_path),
            "outputs": image_paths,
            "elapsed_seconds": round(time.time() - started, 4),
        }
        metadata_path = writer.save_metadata(self.config, metadata)

        return EnvisionResult(
            original_image_path=image_paths["original_image"],
            impression_image_path=image_paths["impression_image"],
            uncertainty_map_path=image_paths["uncertainty_map"],
            uncertainty_heatmap_path=image_paths["uncertainty_heatmap"],
            metadata_path=str(metadata_path),
        )
