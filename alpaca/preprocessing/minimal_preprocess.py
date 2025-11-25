
"""
Minimal preprocessing for ALPaCA when images are already:
- N4 Bias Corrected
- Co-registered
- Skull-stripped  
- Have labeled lesion candidates
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
from pathlib import Path
import shutil

def normalize_image(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalize using precomputed mask"""
    
    mean = data[mask].mean()
    std = data[mask].std()
   
    normalized = np.empty_like(data)
    normalized[mask] = (data[mask] - mean) / std
    normalized[~mask] = 0

    return normalized

def erode_labels(labels: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Vectorized erosion"""

    lesion_mask = labels > 0
    eroded_mask = binary_erosion(lesion_mask, iterations=iterations)

    surviving = eroded_mask & (labels > 0)
    result = np.zeros_like(labels)
    result[surviving] = labels[surviving]
    
    return result


def minimal_preprocess(t1_path: str, flair_path: str, epi_path: str, phase_path: str,
                       labels_path: str, output_dir: str, eroded_candidates_path: str = None,
                    verbose: bool = True):
    """
    Minimal preprocessing: normalize images and erode lesion labels. Assumes 
    already registered and skull stripped. If provided eroded candidates, skips erosion.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n┌" + "─"*38 + "┐")
        print("│" + " Image Preprocessing ".center(38) + "│")
        print("└" + "─"*38 + "┘")   
    
    # 1. Load all images
    if verbose:
        print("[1/4] Loading images...")

    t1_nii = nib.load(t1_path)
    flair_nii = nib.load(flair_path)
    epi_nii = nib.load(epi_path)
    phase_nii = nib.load(phase_path)
    labels_nii = nib.load(labels_path)

    t1 = t1_nii.get_fdata(dtype=np.float32)
    flair = flair_nii.get_fdata(dtype=np.float32)
    epi = epi_nii.get_fdata(dtype=np.float32)
    phase = phase_nii.get_fdata(dtype=np.float32)
    labels = labels_nii.get_fdata().astype(np.int32)
    brain_mask = t1 > 0

    # 2. Normalize images
    if verbose:
        print("[2/4] Normalizing modalities...")
    
    t1_norm = normalize_image(t1, brain_mask)
    flair_norm = normalize_image(flair, brain_mask)
    epi_norm = normalize_image(epi, brain_mask)
    phase_norm = normalize_image(phase, brain_mask)
    
    # 3. Erode labels
    if verbose:
        print("[3/4] Eroding lesion candidates...")

    eroded_out_path = output_dir / "eroded_candidates.nii.gz"

    if eroded_candidates_path is not None:
        eroded = nib.load(eroded_candidates_path).get_fdata(dtype=np.int16)
    else:
        eroded = erode_labels(labels) 

    # 4. [4/4] Saving preprocessed files...
    if verbose:
        print("[4/4] Saving preprocessed files...")

    nib.save(nib.Nifti1Image(t1_norm, t1_nii.affine, t1_nii.header), output_dir / "t1_final.nii.gz")
    nib.save(nib.Nifti1Image(flair_norm, flair_nii.affine, flair_nii.header), output_dir / "flair_final.nii.gz")
    nib.save(nib.Nifti1Image(epi_norm, epi_nii.affine, epi_nii.header), output_dir / "epi_final.nii.gz")
    nib.save(nib.Nifti1Image(phase_norm, phase_nii.affine, phase_nii.header), output_dir / "phase_final.nii.gz")
    nib.save(nib.Nifti1Image(labels, labels_nii.affine, labels_nii.header), output_dir / "labeled_candidates.nii.gz")
    nib.save(nib.Nifti1Image(eroded.astype(np.int16), labels_nii.affine, labels_nii.header), eroded_out_path)

    if verbose:
        print("[Done] Ready for inference.")
    
    return {
        't1': str(output_dir / "t1_final.nii.gz"),
        'flair': str(output_dir / "flair_final.nii.gz"),
        'epi': str(output_dir / "epi_final.nii.gz"),
        'phase': str(output_dir / "phase_final.nii.gz"),
        'labeled_candidates': str(output_dir / "labeled_candidates.nii.gz"),
        'eroded_candidates': str(eroded_out_path)
    }

