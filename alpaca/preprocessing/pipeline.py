"""High-level preprocessing pipelines."""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Optional, Dict

from .normalize import normalize_image
from .erode import erode_labels


def preprocess(
    t1: Union[str, np.ndarray],
    flair: Union[str, np.ndarray],
    epi: Union[str, np.ndarray],
    phase: Union[str, np.ndarray],
    labeled_candidates: Union[str, np.ndarray],
    eroded_candidates: Optional[Union[str, np.ndarray]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Preprocess images for ALPaCA inference.

    Normalizes the 4 MRI modalities and erodes lesion labels.
    Output dict can be unpacked directly into make_predictions().

    Args:
        t1, flair, epi, phase: MRI images (paths or arrays)
        labeled_candidates: Lesion labels (paths or arrays)
        eroded_candidates: Pre-eroded labels (optional, skips erosion if provided)
        output_dir: Optional directory to save preprocessed files

    Returns:
        Dict with keys: t1, flair, epi, phase, labeled_candidates, eroded_candidates

    Example:
        >>> preprocessed = preprocess(t1, flair, epi, phase, labels)
        >>> results = make_predictions(**preprocessed, model_dir='models/')
    """
    # Normalize modalities
    t1_norm = normalize_image(t1)
    flair_norm = normalize_image(flair)
    epi_norm = normalize_image(epi)
    phase_norm = normalize_image(phase)

    # Load labels as array
    if isinstance(labeled_candidates, (str, Path)):
        labels_array = nib.load(str(labeled_candidates)).get_fdata().astype(np.int32)
    else:
        labels_array = np.asarray(labeled_candidates, dtype=np.int32)

    # Erode if not provided
    if eroded_candidates is None:
        eroded = erode_labels(labels_array)
    else:
        if isinstance(eroded_candidates, (str, Path)):
            eroded = nib.load(str(eroded_candidates)).get_fdata().astype(np.int16)
        else:
            eroded = np.asarray(eroded_candidates, dtype=np.int16)

    # Optionally save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Need reference for affine/header
        if isinstance(t1, (str, Path)):
            ref_nii = nib.load(str(t1))
            affine = ref_nii.affine
            header = ref_nii.header
        else:
            affine = np.eye(4)
            header = None

        nib.save(nib.Nifti1Image(t1_norm, affine, header), output_dir / "t1_norm.nii.gz")
        nib.save(nib.Nifti1Image(flair_norm, affine, header), output_dir / "flair_norm.nii.gz")
        nib.save(nib.Nifti1Image(epi_norm, affine, header), output_dir / "epi_norm.nii.gz")
        nib.save(nib.Nifti1Image(phase_norm, affine, header), output_dir / "phase_norm.nii.gz")
        nib.save(nib.Nifti1Image(labels_array, affine, header), output_dir / "labeled_candidates.nii.gz")
        nib.save(nib.Nifti1Image(eroded, affine, header), output_dir / "eroded_candidates.nii.gz")

    return {
        't1': t1_norm,
        'flair': flair_norm,
        'epi': epi_norm,
        'phase': phase_norm,
        'labeled_candidates': labels_array,
        'eroded_candidates': eroded
    }


def run_alpaca(
    t1: Union[str, np.ndarray],
    flair: Union[str, np.ndarray],
    epi: Union[str, np.ndarray],
    phase: Union[str, np.ndarray],
    labeled_candidates: Union[str, np.ndarray],
    eroded_candidates: Optional[Union[str, np.ndarray]] = None,
    model_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    **inference_kwargs
) -> Dict:
    """
    Complete ALPaCA pipeline: preprocessing + inference.

    Args:
        t1, flair, epi, phase: MRI images (paths or arrays)
        labeled_candidates: Lesion labels (paths or arrays)
        eroded_candidates: Pre-eroded labels (optional, skips erosion if provided)
        model_dir: Directory containing model weights
        output_dir: Where to save results
        **inference_kwargs: Additional arguments passed to make_predictions()
            (e.g., n_patches, n_models, lesion_priority, etc.)

    Returns:
        Results dict from make_predictions()

    Example:
        >>> results = run_alpaca(
        ...     t1='t1.nii.gz',
        ...     flair='flair.nii.gz',
        ...     epi='epi.nii.gz',
        ...     phase='phase.nii.gz',
        ...     labeled_candidates='labels.nii.gz',
        ...     model_dir='models/',
        ...     output_dir='results/',
        ...     n_models=10
        ... )
    """
    from ..models.make_predictions import make_predictions

    # Preprocess
    preprocessed = preprocess(
        t1=t1,
        flair=flair,
        epi=epi,
        phase=phase,
        labeled_candidates=labeled_candidates,
        eroded_candidates=eroded_candidates
    )

    # Run inference
    return make_predictions(
        **preprocessed,
        model_dir=model_dir,
        output_dir=output_dir,
        **inference_kwargs
    )
