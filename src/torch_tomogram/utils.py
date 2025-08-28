import torch
import torch_projectors
from torch_grid_utils import fftfreq_grid

# Helper function to crop 3D volumes
def _crop_3d(real_tensor, oversampling_factor):
    current_size = real_tensor.shape[-3]
    original_size = int(current_size / oversampling_factor)
    crop_total = current_size - original_size
    crop_start = crop_total // 2
    crop_end = crop_start + original_size
    return real_tensor[..., crop_start:crop_end, crop_start:crop_end,
    crop_start:crop_end]


def _backproject_2d_to_3d(
        real_projections: torch.Tensor,
        rotations: torch.Tensor,
        pad_factor: int = 2
) -> torch.Tensor:
    n_tilts = real_projections.shape[-3]

    # 1. fftshift and convert to Fourier space
    shifted_projection = torch.fft.fftshift(real_projections, dim=(-2, -1))
    fourier_projection = torch.fft.rfft2(shifted_projection, norm='forward')
    weights = torch.ones_like(
        fourier_projection, dtype=torch.float32, device=fourier_projection.device
    )

    # 2. Set up backprojection parameters (rotation matrix for 3D)
    rotations = torch.flip(rotations, dims=(-2, -1))  # zyx > xyz matrix
    rotations = rotations.unsqueeze(0)  # add batch dimension
    shifts = torch.zeros(1, n_tilts, 2, dtype=torch.float32)

    # 3. Backward project 2D->3D with oversampling=2.0
    data_rec, weight_rec = torch_projectors.backproject_2d_to_3d_forw(
        fourier_projection,  # Add batch dimensions
        rotations,
        weights=weights,
        shifts=shifts,
        interpolation='cubic',
        oversampling=1.0
    )

    # max operation with ones
    weight_rec = torch.clamp(weight_rec, min=1.0)
    data_rec /= weight_rec

    # 4. Convert reconstruction to real space
    real_reconstruction = torch.fft.irfftn(data_rec[0], dim=(-3, -2, -1), norm='forward')
    real_reconstruction = torch.fft.ifftshift(real_reconstruction, dim=(-3, -2, -1))

    # 5. ifftshift and crop to 0.5x size (original size from 2x oversampling)
    result = _crop_3d(real_reconstruction, pad_factor)

    return result