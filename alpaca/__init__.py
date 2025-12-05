"""ALPaCA: Automated Lesion, PRL, and CVS Analysis"""

from .models.make_predictions import make_predictions
from .preprocessing.normalize import normalize_image
from .preprocessing.erode import erode_labels
from .preprocessing.pipeline import preprocess, run_alpaca

__version__ = "1.0.0"
__all__ = [
    "make_predictions",
    "normalize_image",
    "erode_labels",
    "preprocess",
    "run_alpaca",
]
