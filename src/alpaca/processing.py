

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Optional, Dict
from scipy.ndimage import binary_erosion

def normalize_image(
    image: Union[str, np.ndarray],
    mask: np.ndarray,
    output_path: Optional[str] = None
) -> np.ndarray:
    """Z-score normalize an MRI image and mask."""

    ref_nii = None
    if isinstance(image, (str, Path)):
        ref_nii = nib.load(str(image))
        data = ref_nii.get_fdata(dtype=np.float32)
    else:
        data = np.asarray(image, dtype=np.float32)

    # Calculate stats using the boolean mask
    mean = data[mask].mean()
    std = data[mask].std()

    # Normalize
    normalized = np.zeros_like(data)
    if std > 1e-9:
        normalized[mask] = (data[mask] - mean) / std

    # Save if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        affine = ref_nii.affine if ref_nii else np.eye(4)
        header = ref_nii.header if ref_nii else None
        nib.save(nib.Nifti1Image(normalized, affine, header), output_path)

    return normalized

def erode_labels(
    labels: Union[str, np.ndarray],
    iterations: int = 1,
    output_path: Optional[str] = None
) -> np.ndarray:
    """Erode each lesion label independently."""

    # Load if path
    ref_nii = None
    if isinstance(labels, (str, Path)):
        ref_nii = nib.load(str(labels))
        label_data = ref_nii.get_fdata().astype(np.int32)
    else:
        label_data = np.asarray(labels, dtype=np.int32)

    # Erode each lesion independently
    eroded = np.zeros_like(label_data)
    for label_id in np.unique(label_data):
        if label_id == 0:
            continue

        mask = (label_data == label_id)
        eroded_mask = binary_erosion(mask, iterations=iterations)
        eroded[eroded_mask] = label_id

    # Save if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        affine = ref_nii.affine if ref_nii else np.eye(4)
        header = ref_nii.header if ref_nii else None
        nib.save(nib.Nifti1Image(eroded.astype(np.int16), affine, header), output_path)

    return eroded.astype(np.int16)

def preprocess(
    t1: Union[str, np.ndarray],
    flair: Union[str, np.ndarray],
    epi: Union[str, np.ndarray],
    phase: Union[str, np.ndarray],
    labeled_candidates: Union[str, np.ndarray],
    eroded_candidates: Optional[Union[str, np.ndarray]] = None,
    output_dir: Optional[str] = None,
    verbose: Optional[bool] = False
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
        verbose: Verbose output

    Returns:
        Dict with keys: t1, flair, epi, phase, labeled_candidates, eroded_candidates

    Example:
        >>> preprocessed = preprocess(t1, flair, epi, phase, labels)
        >>> results = make_predictions(**preprocessed, model_dir='models/')
    """

    # Normalize modalities
    if verbose:
        print("Normalizing images...")

    if isinstance(t1, (str, Path)):
        t1_data = nib.load(str(t1)).get_fdata()
    else:
        t1_data = np.asarray(t1)

    # Use the mask from T1
    mask = t1_data > 0

    t1_norm = normalize_image(t1, mask)
    flair_norm = normalize_image(flair, mask)
    epi_norm = normalize_image(epi, mask)
    phase_norm = normalize_image(phase, mask)

    # Load labels as array
    if isinstance(labeled_candidates, (str, Path)):
        labels_array = nib.load(str(labeled_candidates)).get_fdata().astype(np.int32)
    else:
        labels_array = np.asarray(labeled_candidates, dtype=np.int32)

    # Erode if not provided
    if eroded_candidates is None:
        if verbose:
            print("Eroding lesions...")
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

        if verbose:
            print(f"Saved processed images to {output_dir}")

        # Return file paths when saving (matches old behavior)
        return {
            't1': str(output_dir / "t1_norm.nii.gz"),
            'flair': str(output_dir / "flair_norm.nii.gz"),
            'epi': str(output_dir / "epi_norm.nii.gz"),
            'phase': str(output_dir / "phase_norm.nii.gz"),
            'labeled_candidates': str(output_dir / "labeled_candidates.nii.gz"),
            'eroded_candidates': str(output_dir / "eroded_candidates.nii.gz")
        }

    # Return arrays when not saving
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
    from .inference import make_predictions

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
