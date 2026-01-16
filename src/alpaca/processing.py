

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Optional, Dict
from scipy.ndimage import binary_erosion

from .logger import log


def _normalize_image(
    image: Union[str, np.ndarray],
    mask: np.ndarray,
    output_path: Optional[str] = None
) -> np.ndarray:
    """Z-score normalize an MRI image with a mask."""

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

    log.info("Eroding lesion candidates...")

    # Load if path
    ref_nii = None
    if isinstance(labels, (str, Path)):
        ref_nii = nib.load(str(labels))
        label_data = ref_nii.get_fdata().astype(np.int32)
    else:
        label_data = np.asarray(labels, dtype=np.int32)

    # Erode each lesion independently
    eroded = np.zeros_like(label_data)
    unique_labels = np.unique(label_data)
    log.debug(f"  - Eroding {len(unique_labels) - 1} labels...")
    for label_id in unique_labels:
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
        log.debug(f"  - Saved eroded labels to {output_path}")

    log.debug("  [green]✓[/green] Erosion complete.")
    return eroded.astype(np.int16)

def normalize_images(
    t1: Union[str, np.ndarray],
    flair: Union[str, np.ndarray],
    epi: Union[str, np.ndarray],
    phase: Union[str, np.ndarray],
    output_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:

    log.info("Normalizing images...")

    # Use the mask from T1
    if isinstance(t1, (str, Path)):
        log.debug("  - Loading T1 from path to create mask...")
        t1_data = nib.load(str(t1)).get_fdata()
    else:
        t1_data = np.asarray(t1)

    mask = t1_data > 0

    log.debug("  - Normalizing T1, FLAIR, EPI, and Phase images...")
    t1_norm = _normalize_image(t1, mask)
    flair_norm = _normalize_image(flair, mask)
    epi_norm = _normalize_image(epi, mask)
    phase_norm = _normalize_image(phase, mask)
    
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

    log.debug("  [green]✓[/green] Normalization complete.")

    # Return arrays when not saving
    return {
        't1': t1_norm,
        'flair': flair_norm,
        'epi': epi_norm,
        'phase': phase_norm
    }

from .inference import make_predictions


def run_alpaca(
    t1: Union[str, np.ndarray],
    flair: Union[str, np.ndarray],
    epi: Union[str, np.ndarray],
    phase: Union[str, np.ndarray],
    labeled_candidates: Union[str, np.ndarray],
    eroded_candidates: Optional[Union[str, np.ndarray]] = None,
    skip_normalization: Optional[bool] = False,
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
        skip_normalization: Skip normalization step
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
        ...     batch_size=20
        ... )
    """
    # Normalize
    if not skip_normalization:
        normalized = normalize_images(
            t1=t1,
            flair=flair,
            epi=epi,
            phase=phase
        )

    else:
        log.info("[yellow]Skipping normalization.[/yellow]")
        normalized = {
            't1': t1,
            'flair': flair,
            'epi': epi,
            'phase': phase
        }

    # Load labels as array
    if isinstance(labeled_candidates, (str, Path)):
        labels_array = nib.load(str(labeled_candidates)).get_fdata().astype(np.int32)
    else:
        labels_array = np.asarray(labeled_candidates, dtype=np.int32)

    # Erode if not provided
    if eroded_candidates is None:
        eroded = erode_labels(labels_array)
    else:
        log.info("[yellow]Using pre-eroded labels.[/yellow]")
        if isinstance(eroded_candidates, (str, Path)):
            eroded = nib.load(str(eroded_candidates)).get_fdata().astype(np.int16)
        else:
            eroded = np.asarray(eroded_candidates, dtype=np.int16)

    # Run inference
    results = make_predictions(
        **normalized,
        labeled_candidates=labeled_candidates,
        eroded_candidates=eroded,
        model_dir=model_dir,
        output_dir=output_dir,
        **inference_kwargs
    )
    
    return results
