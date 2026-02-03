

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Optional, Dict
from scipy.ndimage import binary_erosion
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage

from .logger import log
from .inference import make_predictions

def label_lesions(
        prob_map: Union[str, np.ndarray], 
        threshold: float = 0.05, 
        sigma: float = 1,
        output_dir: Optional[str] = None):
    """Apply threshold to lesion probability map and split confluent lesions, enumerate with labels""" 

    log.info("Splitting confluent lesions...") 
    
    # Load probability map
    ref_nii = None
    if isinstance(prob_map, (str, Path)):
        ref_nii = nib.load(str(prob_map))
        prob_map = ref_nii.get_fdata(dtype=np.float32)
    else:
        prob_map = np.asarray(prob_map, dtype=np.float32)

    # 1. Create the Binary Mask
    binary_mask = prob_map > threshold
    if not np.any(binary_mask):
        return np.zeros_like(prob_map, dtype=int)

    # 2. Calculate Hessian Eigenvalues
    hxx, hxy, hxz, hyy, hyz, hzz = hessian_matrix(prob_map, sigma=sigma)
    i1, i2, i3 = hessian_matrix_eigvals([hxx, hxy, hxz, hyy, hyz, hzz])

    # 3. Find Lesion Centers
    potential_centers = (i1 < 0) & (i2 < 0) & (i3 < 0)  # Centers go down (negative)
    centers_mask = potential_centers & binary_mask
    markers, num_centers = label(centers_mask, return_num=True)
    log.debug(f"  - {num_centers} lesions labeled.")

    # 4. Watershed Separation
    split_labels = watershed(-prob_map, markers, mask=binary_mask)

    # Save if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        affine = ref_nii.affine if ref_nii else np.eye(4)
        header = ref_nii.header if ref_nii else None
        save_path = output_dir / "labeled_candidates.nii.gz"
        nib.save(nib.Nifti1Image(split_labels.astype(np.int16), affine, header), save_path)
        log.debug(f"  - Saved candidates to {save_path}.")

    log.debug("  [green]✓[/green] Labeling complete.") 
 
    return split_labels

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

def normalize_images(
    t1: Union[str, np.ndarray],
    flair: Union[str, np.ndarray],
    epi: Union[str, np.ndarray],
    phase: Union[str, np.ndarray],
    output_path: Optional[str] = None,
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
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Need reference for affine/header
        if isinstance(t1, (str, Path)):
            ref_nii = nib.load(str(t1))
            affine = ref_nii.affine
            header = ref_nii.header
        else:
            affine = np.eye(4)
            header = None

        nib.save(nib.Nifti1Image(t1_norm, affine, header), output_path / "t1_norm.nii.gz")
        nib.save(nib.Nifti1Image(flair_norm, affine, header), output_path / "flair_norm.nii.gz")
        nib.save(nib.Nifti1Image(epi_norm, affine, header), output_path / "epi_norm.nii.gz")
        nib.save(nib.Nifti1Image(phase_norm, affine, header), output_path / "phase_norm.nii.gz")
        log.debug(f"  - Saved images to {output_path}.")

    log.debug("  [green]✓[/green] Normalization complete.")

    # Return arrays when not saving
    return {
        't1': t1_norm,
        'flair': flair_norm,
        'epi': epi_norm,
        'phase': phase_norm
    }

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

def run_alpaca(
    t1: Union[str, np.ndarray],
    flair: Union[str, np.ndarray],
    epi: Union[str, np.ndarray],
    phase: Union[str, np.ndarray],
    prob_map: Optional[Union[str, np.ndarray]] = None,
    labeled_candidates: Optional[Union[str, np.ndarray]] = None,
    eroded_candidates: Optional[Union[str, np.ndarray]] = None,
    skip_normalization: Optional[bool] = False,
    candidate_threshold: Optional[float] = 0.05,
    model_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    
    **inference_kwargs
) -> Dict:
    """
    Complete ALPaCA pipeline: preprocessing + inference.

    Args:
        t1, flair, epi, phase: MRI images (paths or arrays)
        prob_map: Lesion probability map (path or array, provide this or labeled_candidates)
        labeled_candidates: Lesion labels (path or array, provide this or prob_map)
        eroded_candidates: Pre-eroded labels (optional, skips erosion if provided)
        skip_normalization: Skip normalization step
        candidate_threshold: Optional threshold for splitting lesions from probability map (default: 0.05)
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
        ...     prob_map='prob_map.nii.gz',
        ...     model_dir='models/',
        ...     output_dir='results/',
        ...     batch_size=20
        ... )
    """

    if labeled_candidates is None and prob_map is None:
        raise ValueError("Either `labeled_candidates` or `prob_map` must be provided.")

    # Normalize images
    if not skip_normalization:
        normalized = normalize_images(t1=t1, flair=flair, epi=epi, phase=phase)
    else:
        log.info("[yellow]Skipping normalization.[/yellow]")
        normalized = {
            't1': np.asarray(nib.load(str(t1)).get_fdata()) if isinstance(t1, (str, Path)) else np.asarray(t1),
            'flair': np.asarray(nib.load(str(flair)).get_fdata()) if isinstance(flair, (str, Path)) else np.asarray(flair),
            'epi': np.asarray(nib.load(str(epi)).get_fdata()) if isinstance(epi, (str, Path)) else np.asarray(epi),
            'phase': np.asarray(nib.load(str(phase)).get_fdata()) if isinstance(phase, (str, Path)) else np.asarray(phase),
        }

    # If labeled candidates provided, skip labeling
    if labeled_candidates:
        log.info("[yellow]Skipping labeling.[/yellow]")
        if isinstance(labeled_candidates, (str, Path)):
            labels = nib.load(str(labeled_candidates)).get_fdata().astype(np.int32)
        else:
            labels = np.asarray(labeled_candidates, dtype=np.int32)
    else:
        labels = label_lesions(prob_map=prob_map, threshold=candidate_threshold)

    # Erode if not provided
    if eroded_candidates is None:
        eroded = erode_labels(labels)
    else:
        log.info("[yellow]Using pre-eroded labels.[/yellow]")
        if isinstance(eroded_candidates, (str, Path)):
            eroded = nib.load(str(eroded_candidates)).get_fdata().astype(np.int16)
        else:
            eroded = np.asarray(eroded_candidates, dtype=np.int16)

    # Run inference
    results = make_predictions(
        **normalized,
        labeled_candidates=labels,
        eroded_candidates=eroded,
        model_dir=model_dir,
        output_dir=output_dir,
        **inference_kwargs
    )
    
    return results
