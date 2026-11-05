"""Optional pretrained SigLIP visual feature and zero-shot scoring adapter.

The model checkpoint is not bundled with ADA. It is loaded from the local
Hugging Face cache or a user-supplied local model directory by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from ..contracts import Modality, Observation, PerceptionInput, PerceptionStatus, content_digest
from .adapter import VisualAdapterCapabilities
from .preprocessing import DecodedImage, decode_image


DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_SIGLIP_REVISION = "main"


@dataclass(frozen=True, slots=True)
class ZeroShotLabelScore:
    """Sigmoid-transformed image/text matching score, not calibrated confidence."""

    label: str
    sigmoid_score: float


class SigLIPVisionAdapter:
    """Image encoder using Google's pretrained SigLIP Base vision tower.

    perceive() always extracts a normalized image embedding. When the input
    metadata includes candidate_labels, it additionally computes SigLIP
    image/text matching scores for those caller-provided labels.

    Scores are ranking signals, not calibrated probabilities, and labels are
    not an exhaustive description of image contents.
    """

    modality = Modality.IMAGE
    capabilities = VisualAdapterCapabilities(
        supports_images=True,
        supports_video_frames=False,
        supports_ocr=False,
        supports_object_detection=False,
        supports_spatial_grounding=False,
        supports_image_embeddings=True,
        supports_zero_shot_label_scoring=True,
    )

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        model_id: str,
        revision: str,
        device: str,
        torch_module: Any | None = None,
        max_image_pixels: int = 40_000_000,
    ) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        if not callable(getattr(model, "get_image_features", None)):
            raise TypeError("model must expose get_image_features(pixel_values=...)")
        if not callable(model):
            raise TypeError("model must be callable for image/text scoring")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be non-empty")
        if type(max_image_pixels) is not int or max_image_pixels < 1:
            raise ValueError("max_image_pixels must be a positive integer")

        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise ImportError(
                    "SigLIP requires the optional perception-vision dependencies"
                ) from exc

        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.requested_revision = revision
        resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
        self.resolved_revision = resolved or revision
        self.device = device
        self._torch = torch_module
        self.max_image_pixels = max_image_pixels
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_SIGLIP_MODEL,
        *,
        revision: str = DEFAULT_SIGLIP_REVISION,
        device: str | None = None,
        local_files_only: bool = True,
        max_image_pixels: int = 40_000_000,
    ) -> "SigLIPVisionAdapter":
        """Load a Hugging Face checkpoint without enabling repository code.

        Set local_files_only=False only when you explicitly want Transformers
        to download the selected model revision.
        """
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Install ADA's optional visual dependencies: "
                "pip install -e '.[perception-vision]'"
            ) from exc

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = {
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        model = AutoModel.from_pretrained(model_id, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        model.to(target_device)
        return cls(
            model=model,
            processor=processor,
            model_id=model_id,
            revision=revision,
            device=target_device,
            torch_module=torch,
            max_image_pixels=max_image_pixels,
        )

    def _decode(self, item: PerceptionInput) -> DecodedImage:
        if item.modality is not Modality.IMAGE:
            raise TypeError("SigLIPVisionAdapter only accepts IMAGE inputs")
        if not isinstance(item.content, bytes):
            raise TypeError("IMAGE input content must be bytes")
        return decode_image(
            item.content,
            item.content_type,
            max_pixels=self.max_image_pixels,
        )

    def _move_batch(self, batch: Any) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if callable(getattr(value, "to", None)) else value
        return moved

    def _extract_features(self, pixel_values: Any) -> Any:
        torch = self._torch
        with torch.inference_mode():
            output = self.model.get_image_features(pixel_values=pixel_values)

        if isinstance(output, torch.Tensor):
            features = output
        elif getattr(output, "pooler_output", None) is not None:
            features = output.pooler_output
        elif getattr(output, "last_hidden_state", None) is not None:
            features = output.last_hidden_state[:, 0]
        elif isinstance(output, (tuple, list)) and output:
            features = output[0]
        else:
            raise RuntimeError("SigLIP returned no recognized image feature tensor")

        if not isinstance(features, torch.Tensor):
            raise TypeError("SigLIP image features must be a tensor")
        if features.ndim == 3:
            features = features[:, 0, :]
        if features.ndim != 2 or features.shape[0] != 1:
            raise ValueError(f"unexpected SigLIP feature shape: {tuple(features.shape)}")
        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        if not torch.isfinite(features).all().item():
            raise ValueError("SigLIP produced non-finite image features")
        return features[0].detach().cpu()

    def _score_labels(
        self, image: Any, labels: Sequence[str]
    ) -> tuple[ZeroShotLabelScore, ...]:
        if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
            raise TypeError("candidate_labels must be a sequence of strings")
        if not 1 <= len(labels) <= 100:
            raise ValueError("candidate_labels must contain between 1 and 100 labels")
        normalized: list[str] = []
        seen: set[str] = set()
        for label in labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("every candidate label must be a non-empty string")
            clean = label.strip()
            key = clean.casefold()
            if key in seen:
                raise ValueError(f"duplicate candidate label: {clean}")
            seen.add(key)
            normalized.append(clean)

        prompts = [f"a photo of {label}" for label in normalized]
        batch = self.processor(
            text=prompts,
            images=image,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = self._move_batch(batch)
        with self._torch.inference_mode():
            output = self.model(**inputs)
        logits = getattr(output, "logits_per_image", None)
        if logits is None or logits.ndim != 2 or tuple(logits.shape) != (1, len(normalized)):
            raise RuntimeError("SigLIP model did not return expected logits_per_image")
        scores = self._torch.sigmoid(logits.float())[0].detach().cpu().tolist()
        result = [
            ZeroShotLabelScore(label=label, sigmoid_score=float(score))
            for label, score in zip(normalized, scores, strict=True)
        ]
        return tuple(sorted(result, key=lambda item: item.sigmoid_score, reverse=True))

    def perceive(self, item: PerceptionInput) -> Observation:
        decoded = self._decode(item)
        image_batch = self.processor(images=decoded.image, return_tensors="pt")
        pixel_values = self._move_batch(image_batch)["pixel_values"]
        embedding = self._extract_features(pixel_values)
        embedding_values = [float(value) for value in embedding.tolist()]
        if not embedding_values or any(not math.isfinite(value) for value in embedding_values):
            raise ValueError("SigLIP embedding is empty or non-finite")

        attributes: dict[str, Any] = {
            "model_family": "SigLIP",
            "model_id": self.model_id,
            "model_revision": self.resolved_revision,
            "requested_revision": self.requested_revision,
            "embedding_type": "l2_normalized_image_feature",
            "embedding_dimension": len(embedding_values),
            "embedding": embedding_values,
            "image_format": decoded.detected_format,
            "original_width": decoded.original_width,
            "original_height": decoded.original_height,
            "oriented_width": decoded.oriented_width,
            "oriented_height": decoded.oriented_height,
            "preprocessing": "model AutoProcessor image resize/rescale/normalization",
        }

        candidate_labels = item.metadata.get("candidate_labels")
        warnings = [
            "Image embedding is a learned representation, not a caption or object detection result.",
            "Sigmoid label scores are not calibrated probabilities and depend on the caller's candidate labels.",
        ]
        if candidate_labels is not None:
            scores = self._score_labels(decoded.image, candidate_labels)
            attributes["candidate_scores"] = [
                {"label": score.label, "sigmoid_score": score.sigmoid_score}
                for score in scores
            ]
            attributes["candidate_score_semantics"] = (
                "sigmoid-transformed image/text logits; not calibrated probabilities"
            )

        return Observation(
            modality=Modality.IMAGE,
            status=PerceptionStatus.OBSERVED,
            summary=f"SigLIP extracted a {len(embedding_values)}-dimensional image embedding",
            source_input_id=item.input_id,
            content_sha256=content_digest(item.content),
            attributes=attributes,
            warnings=tuple(warnings),
        )
