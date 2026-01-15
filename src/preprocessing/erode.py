"""Morphological erosion for lesion labels."""

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
from pathlib import Path
from typing import Union, Optional


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
