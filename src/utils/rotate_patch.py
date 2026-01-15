
# rotate_patch.py

import torch

def rotate_patch(patch, invert, face, rotations):
    """
    Applies 3D rotation to a patch tensor.

    Args:
        patch: torch.Tensor of shape [D, H, W] (3D volume)
        invert: int (0 or 1) - whether to mirror the patch
        face: int (1-6) - which face of the cube to orient "down"
        rotations: int (0-3) - number of 90 degree rotations around vertical axis

    Returns:
        torch.Tensor of same shape, rotated

    Note:
        Total: 2 x 6 x 4 = 48 possible orientations
        Slows down inference
    """

    # Step 1: Optional reflection (mirror along first axis)
    if invert == 1:
        patch = torch.flip(patch, dims=[0])


    # Step 2: Reorient which face is "down"    
    if face == 2:     # Flip cube towards you
        patch = torch.flip(patch, dims=[0])
        patch = patch.transpose(0, 1)

    elif face == 3:     # Flip cube towards you twice
        patch = torch.flip(patch, dims=[0])
        patch = patch.transpose(0, 1)
        patch = torch.flip(patch, dims=[0])
        patch = patch.transpose(0, 1)

    elif face == 4:     # Flip cube away from you
        patch = torch.flip(patch, dims=[1])
        patch = patch.transpose(1, 0)

    elif face == 5:     # Flip cube to the left
        patch = torch.flip(patch, dims=[2])
        patch = patch.transpose(0, 2)

    elif face == 6:     # Flip cube to the right
        patch = torch.flip(patch, dims=[0])
        patch = patch.transpose(2, 0)


    # Step 3: Rotate around vertical axis (radial rotation)
    if rotations == 1:  # 90 clockwise
        patch = torch.flip(patch, dims=[1])
        patch = patch.transpose(1, 2)

    if rotations == 2:  # 180
        patch = torch.flip(patch, dims=[1])
        patch = patch.transpose(1, 2)
        patch = torch.flip(patch, dims=[1])
        patch = patch.transpose(1, 2)

    if rotations == 3:  # 270 clockwise
        patch = torch.flip(patch, dims=[2])
        patch = patch.transpose(2, 1)

    return patch
