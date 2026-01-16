"""ALPaCA: Automated Lesion, PRL, and CVS Analysis"""

from .inference import make_predictions
from .processing import normalize_image, erode_labels, preprocess, run_alpaca

__version__ = "1.0.0"
__all__ = [
    "make_predictions",
    "normalize_image",
    "erode_labels",
    "preprocess",
    "run_alpaca",
]
