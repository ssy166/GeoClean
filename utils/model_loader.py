import os

import torch


class ModelManager:
    """Minimal public loader for the verified FLUX GeoClean path."""

    def __init__(
        self,
        model_id="./models/FLUX.1-dev",
        device="cuda",
    ):
        if not os.path.exists(model_id):
            print(f"Warning: model path does not exist yet: {model_id}")

        self.model_id = model_id
        self.device = device
        self.pipeline = None

    def load_pipeline(self):
        if self.pipeline is None:
            from diffusers import FluxPipeline

            print(f"Loading FLUX pipeline from local path: {self.model_id}")
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to(self.device)
        return self.pipeline
