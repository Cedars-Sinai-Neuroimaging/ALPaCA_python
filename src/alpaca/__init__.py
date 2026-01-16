"""ALPaCA: Automated Lesion, PRL, and CVS Analysis"""

from .inference import make_predictions
from .processing import erode_labels, normalize_images, run_alpaca

__version__ = "1.0.0"
__all__ = [
    "make_predictions",
    "erode_labels",
    "normalize_images",
    "run_alpaca",
]
