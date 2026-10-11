from typing import Dict, Any, Optional
from ..validation.olofsson import estimate_area_with_reference_samples as olofsson_estimate

def estimate_area_with_reference_samples(
    classification_map: Any,
    reference_samples: Optional[Any] = None,
    strata_definition: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Interface/hook for Olofsson area estimation using independent reference samples.
    """
    return olofsson_estimate(
        classification_map=classification_map,
        reference_samples=reference_samples,
        strata_definition=strata_definition,
        config=config
    )

