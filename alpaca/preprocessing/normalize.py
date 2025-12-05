"""Image intensity normalization."""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Optional


def normalize_image(
    image: Union[str, np.ndarray],
    output_path: Optional[str] = None
) -> np.ndarray:
    """Z-score normalize an MRI image."""

    # Load if path
    ref_nii = None
    if isinstance(image, (str, Path)):
        ref_nii = nib.load(str(image))
        data = ref_nii.get_fdata(dtype=np.float32)
    else:
        data = np.asarray(image, dtype=np.float32)

    # Check if already normalized (use abs to include negative values)
    check_mask = (np.abs(data) > 1e-6)
    if check_mask.sum() > 0:
        mean_check = data[check_mask].mean()
        std_check = data[check_mask].std()
        if abs(mean_check) < 0.01 and abs(std_check - 1.0) < 0.01:
            return data

    # Normalize using positive voxels
    mask = (data > 0)
    mean = data[mask].mean()
    std = data[mask].std()

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (data[mask] - mean) / std

    # Save if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        affine = ref_nii.affine if ref_nii else np.eye(4)
        header = ref_nii.header if ref_nii else None
        nib.save(nib.Nifti1Image(normalized, affine, header), output_path)

    return normalized
