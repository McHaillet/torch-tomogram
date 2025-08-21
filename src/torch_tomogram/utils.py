import torch
import torch_projectors

# Helper function to crop 3D volumes
def _ifftshift_and_crop_3d(real_tensor, oversampling_factor):
    shifted = torch.fft.ifftshift(real_tensor, dim=(-3, -2, -1))
    current_size = real_tensor.shape[-3]
    original_size = int(current_size / oversampling_factor)
    crop_total = current_size - original_size
    crop_start = crop_total // 2
    crop_end = crop_start + original_size
    return shifted[..., crop_start:crop_end, crop_start:crop_end, crop_start:crop_end]


def _backproject_2d_to_3d(
        real_projections: torch.Tensor,
        rotations: torch.Tensor,
        pad_factor: int = 2
) -> torch.Tensor:
    n_tilts = real_projections.shape[-3]

    # 1. fftshift and convert to Fourier space
    shifted_projection = torch.fft.fftshift(real_projections, dim=(-2, -1))
    fourier_projection = torch.fft.rfft2(shifted_projection, norm='forward')

    # 2. Set up backprojection parameters (rotation matrix for 3D)
    rotations = torch.flip(rotations, dims=(-2, -1))  # zyx > xyz matrix
    rotations = rotations.unsqueeze(0)  # add batch dimension
    shifts = torch.zeros(1, n_tilts, 2, dtype=torch.float32)

    # 3. Backward project 2D->3D with oversampling=2.0
    data_rec, _ = torch_projectors.backproject_2d_to_3d_forw(
        fourier_projection,  # Add batch dimensions
        rotations,
        shifts=shifts,
        interpolation='linear',
        oversampling=1.0
    )

    # 4. Convert reconstruction to real space
    real_reconstruction = torch.fft.irfftn(data_rec[0], dim=(-3, -2, -1), norm='forward')

    # 5. ifftshift and crop to 0.5x size (original size from 2x oversampling)
    result = _ifftshift_and_crop_3d(real_reconstruction, pad_factor)

    return result