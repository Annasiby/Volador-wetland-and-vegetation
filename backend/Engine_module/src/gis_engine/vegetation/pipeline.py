from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio

from ..core.constants import NODATA_CATEGORICAL
from ..core.exceptions import InferenceError
from ..inference.model import SingleDateInferenceEngine
from ..inference.probabilities import calculate_softmax, calculate_confidence


CLASS_NAMES = {
    0: "Background / Bare Soil",
    1: "Severe Stress",
    2: "Moderate Stress",
    3: "Healthy Vegetation",
}


def calculate_class_stats(
    class_map: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    total_valid = int(np.sum(mask))

    stats = {
        "total_valid_pixels": total_valid,
        "classes": {},
    }

    class_label_map = {
        0: "Background",
        1: "Severe",
        2: "Moderate",
        3: "Healthy",
    }

    for class_id, name in class_label_map.items():
        if total_valid > 0:
            count = int(np.sum((class_map == class_id) & mask))
            pct = (count / total_valid) * 100.0
        else:
            count = 0
            pct = 0.0

        stats["classes"][str(class_id)] = {
            "name": name,
            "count": count,
            "percentage": float(pct),
        }

    return stats


@dataclass
class SingleDateVegetationResult:
    health_map: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    metadata: Any
    class_statistics: Dict[str, Any]
    performance: Dict[str, Any]


class VegetationAnalysisPipeline:
    """
    Single-date vegetation-health analysis.

    This pipeline intentionally performs ONLY:
        scene -> preprocessing -> inference -> health/stress metrics

    Temporal operations such as:
        - T1/T2 comparison
        - transition analysis
        - CVA
        - degradation
        - recovery

    belong to the wetland temporal pipeline and are not performed here.
    """

    def run(
        self,
        *,
        scene_dir: Path,
        model_path: Path,
        veg_dir: Path,
        config: Dict[str, Any],
        run_logger: Any,
    ) -> SingleDateVegetationResult:

        scene_dir = Path(scene_dir)
        model_path = Path(model_path)
        veg_dir = Path(veg_dir)

        veg_dir.mkdir(parents=True, exist_ok=True)

        run_logger.info(
            "Vegetation analysis — single-date inference"
        )

        # --------------------------------------------------------------
        # 1. Initialize the existing Phase-1 single-date engine
        # --------------------------------------------------------------
        config = dict(config)
        config["model"] = dict(config["model"])
        config["model"]["path"] = str(model_path)

        try:
            engine = SingleDateInferenceEngine(config)

            logits, class_map, metadata, run_stats = (
                engine.run_pipeline(scene_dir)
            )

        except Exception as exc:
            raise InferenceError(
                f"Single-date vegetation inference failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # 2. Build valid-data mask
        # --------------------------------------------------------------
        valid_mask = np.ones(
            (metadata.height, metadata.width),
            dtype=bool,
        )

        # The existing inference engine has already validated the bands.
        # Here we derive a conservative validity mask from finite values
        # in the model output.
        valid_mask &= np.all(
            np.isfinite(logits),
            axis=(0, 1),
        )

        health_map = np.where(
            valid_mask,
            class_map,
            255,
        ).astype(np.uint8)

        # --------------------------------------------------------------
        # 3. Probabilities and confidence
        # --------------------------------------------------------------
        probabilities = calculate_softmax(logits)
        confidence = calculate_confidence(probabilities)

        confidence = np.where(
            valid_mask,
            confidence,
            0.0,
        )

        # --------------------------------------------------------------
        # 4. Vegetation statistics
        # --------------------------------------------------------------
        class_statistics = calculate_class_stats(
            class_map,
            valid_mask,
        )

        # --------------------------------------------------------------
        # 5. Write health map
        # --------------------------------------------------------------
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "nodata": NODATA_CATEGORICAL,
            "width": metadata.width,
            "height": metadata.height,
            "count": 1,
            "crs": metadata.crs,
            "transform": metadata.transform,
            "compress": "lzw",
        }

        with rasterio.open(
            veg_dir / "health.tif",
            "w",
            **profile,
        ) as dst:
            dst.write(health_map, 1)

        # --------------------------------------------------------------
        # 6. Write confidence map
        # --------------------------------------------------------------
        confidence_profile = {
            **profile,
            "dtype": "float32",
            "nodata": 0.0,
        }

        with rasterio.open(
            veg_dir / "confidence.tif",
            "w",
            **confidence_profile,
        ) as dst:
            dst.write(
                confidence.astype(np.float32),
                1,
            )

        # --------------------------------------------------------------
        # 7. Save raw model outputs
        # --------------------------------------------------------------
        np.save(
            veg_dir / "logits.npy",
            logits,
        )

        np.save(
            veg_dir / "probabilities.npy",
            probabilities,
        )

        # --------------------------------------------------------------
        # 8. Write statistics
        # --------------------------------------------------------------
        import json

        statistics = {
            "analysis_type": "single_date_vegetation",
            "class_statistics": class_statistics,
            "performance": run_stats,
            "model_input_shape": [
                1,
                5,
                metadata.height,
                metadata.width,
            ],
            "model_output_shape": [
                1,
                4,
                metadata.height,
                metadata.width,
            ],
            "class_mapping": CLASS_NAMES,
        }

        with open(
            veg_dir / "statistics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                statistics,
                f,
                indent=2,
            )

        # --------------------------------------------------------------
        # 9. Register outputs
        # --------------------------------------------------------------
        run_logger.info(
            "Vegetation analysis — single-date inference complete"
        )

        return SingleDateVegetationResult(
            health_map=health_map,
            logits=logits,
            probabilities=probabilities,
            confidence=confidence,
            valid_mask=valid_mask,
            metadata=metadata,
            class_statistics=class_statistics,
            performance=run_stats,
        )
