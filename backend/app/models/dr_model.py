from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class DRModel:

    def __init__(self, weights_path: Path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.weights_path = weights_path

        self.model = self._build_model()

        self._load_weights()

        self.model.to(self.device)
        self.model.eval()

        # ====================================================
        # Grad-CAM
        # ====================================================

        self.target_layer = self.model.features[-1]

        self.activations = None
        self.gradients = None

        self._register_gradcam_hooks()

        print(f"DR Model loaded on: {self.device}")


    # ========================================================
    # BUILD MODEL
    # ========================================================

    def _build_model(self):

        model = models.efficientnet_b7(
            weights=None
        )

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            5
        )

        return model


    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    def _load_weights(self):

        if not self.weights_path.exists():

            raise FileNotFoundError(
                f"DR model weights not found: "
                f"{self.weights_path}"
            )

        checkpoint = torch.load(
            self.weights_path,
            map_location=self.device
        )

        state_dict = checkpoint

        clean_state_dict = {}

        for key, value in state_dict.items():

            key = key.replace(
                "module.",
                ""
            )

            key = key.replace(
                "model.",
                ""
            )

            clean_state_dict[key] = value

        self.model.load_state_dict(
            clean_state_dict,
            strict=True
        )

        print(
            f"Loaded DR weights: "
            f"{self.weights_path.name}"
        )


    # ========================================================
    # GRAD-CAM HOOKS
    # ========================================================

    def _register_gradcam_hooks(self):

        def forward_hook(
            module,
            input,
            output
        ):

            self.activations = output

        def backward_hook(
            module,
            grad_input,
            grad_output
        ):

            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_full_backward_hook(
            backward_hook
        )


    # ========================================================
    # PREDICTION + GRAD-CAM
    # ========================================================

    def predict_with_gradcam(
        self,
        input_tensor,
        original_image
    ):

        input_tensor = input_tensor.to(
            self.device
        )

        input_tensor.requires_grad_(True)

        self.model.zero_grad()

        self.activations = None
        self.gradients = None

        logits = self.model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )

        confidence = float(
            probabilities[
                predicted_class
            ].item()
        )

        probability_list = [
            float(value)
            for value in probabilities.detach()
            .cpu()
            .numpy()
        ]

        logits[
            0,
            predicted_class
        ].backward()

        if (
            self.gradients is None
            or self.activations is None
        ):

            raise RuntimeError(
                "Grad-CAM activations or gradients "
                "were not captured."
            )

        pooled_gradients = torch.mean(
            self.gradients,
            dim=[0, 2, 3]
        )

        activations = self.activations[
            0
        ].detach()

        for channel in range(
            activations.shape[0]
        ):

            activations[channel] *= (
                pooled_gradients[channel]
            )

        cam = torch.mean(
            activations,
            dim=0
        )

        cam = torch.relu(cam)

        cam = cam.cpu().numpy()

        if np.max(cam) != 0:

            cam = (
                cam
                / np.max(cam)
            )

        original_height, original_width = (
            original_image.shape[:2]
        )

        cam = cv2.resize(
            cam,
            (
                original_width,
                original_height
            )
        )

        cam_uint8 = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            cam_uint8,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            original_image,
            0.65,
            heatmap,
            0.35,
            0
        )

        return {
            "class_index": predicted_class,
            "confidence": confidence,
            "probabilities": probability_list,
            "heatmap": overlay
        }
