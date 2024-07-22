"""This module provides functionality to perform demons registration using SimpleITK."""

from __future__ import annotations

from typing import Optional

import SimpleITK as sitk
import torch

from direct.registration.simpleitk_utils import convert_to_sitk_image, convert_to_tensor
from direct.types import DirectEnum, TensorOrNdarray


class DemonsFilterType(DirectEnum):
    DEMONS = "demons"
    FAST_SYMMETRIC_FORCES = "fast_symmetric_forces"
    SYMMETRIC_FORCES = "symmetric_forces"
    DIFFEOMORPHIC = "diffeomorphic"


def create_demons_filter(
    filter_type: DemonsFilterType = DemonsFilterType.DEMONS,
    num_iterations: int = 100,
    smooth_displacement_field: bool = True,
    standard_deviations: float = 1.5,
    intensity_difference_threshold: Optional[float] = None,
    maximum_rms_error: Optional[float] = None,
) -> sitk.DemonsRegistrationFilter:
    """Create and configure a Demons filter.

    Parameters
    ----------
    filter_type : DemonsFilterType
        Type of the Demons filter (DemonsFilterType.DEMONS, DemonsFilterType.FAST_SYMMETRIC_FORCES,
        DemonsFilterType.SYMMETRIC_FORCES, DemonsFilterType.DIFFEOMORPHIC). Default: DemonsFilterType.DEMONS.
    num_iterations : int
        Number of iterations for the Demons filter. Default: 100.
    smooth_displacement_field : bool
        Whether to smooth the displacement field. Default: True.
    standard_deviations : float
        Standard deviations for Gaussian smoothing. Default: 1.5.
    intensity_difference_threshold : float, optional
        Intensity difference threshold. Default: None.
    maximum_rms_error : float, optional
        Maximum RMS error. Default: None.

    Returns
    -------
    sitk.DemonsRegistrationFilter
        Configured Demons filter.
    """
    if filter_type == DemonsFilterType.SYMMETRIC_FORCES:
        demons_filter = sitk.SymmetricForcesDemonsRegistrationFilter()
    elif filter_type == DemonsFilterType.DIFFEOMORPHIC:
        demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()
    elif filter_type == DemonsFilterType.FAST_SYMMETRIC_FORCES:
        demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    else:
        demons_filter = sitk.DemonsRegistrationFilter()

    demons_filter.SetNumberOfIterations(num_iterations)
    demons_filter.SetSmoothDisplacementField(smooth_displacement_field)
    demons_filter.SetStandardDeviations(standard_deviations)

    if intensity_difference_threshold is not None:
        demons_filter.SetIntensityDifferenceThreshold(intensity_difference_threshold)

    if maximum_rms_error is not None:
        demons_filter.SetMaximumRMSError(maximum_rms_error)

    return demons_filter


def simpleitk_multiscale_demons_registration(
    reference_image: TensorOrNdarray,
    moving_image: TensorOrNdarray,
    filter_type: DemonsFilterType = DemonsFilterType.DEMONS,
    num_iterations: int = 100,
    smooth_displacement_field: bool = True,
    standard_deviations: float = 1.5,
    intensity_difference_threshold: Optional[float] = None,
    maximum_rms_error: Optional[float] = None,
) -> torch.Tensor:
    """Perform multiscale demons registration using SimpleITK.

    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.

    Parameters
    ----------
    reference_image : torch.Tensor or np.ndarray
        Reference image to register to of shape (H, W) or (D, H, W).
    moving_image : torch.Tensor or np.ndarray
        Moving image to register of shape (H, W) or (D, H, W).
    filter_type : DemonsFilterType, optional
        Type of the Demons filter (DemonsFilterType.DEMONS, DemonsFilterType.FAST_SYMMETRIC_FORCES,
        DemonsFilterType.SYMMETRIC_FORCES, DemonsFilterType.DIFFEOMORPHIC). Default: DemonsFilterType.DEMONS.
    num_iterations : int
        Number of iterations for the Demons filter. Default: 100.
    smooth_displacement_field : bool
        Whether to smooth the displacement field. Default: True.
    standard_deviations : float
        Standard deviations for Gaussian smoothing. Default: 1.5.
    intensity_difference_threshold : float, optional
        Intensity difference threshold. Default: None.
    maximum_rms_error : float, optional
        Maximum RMS error. Default: None.

    Returns
    -------
    torch.Tensor
        Displacement field tensor of shape (H, W, 2) or (D, H, W, 3).
    """

    # Create the registration algorithm
    registration_algorithm = create_demons_filter(
        filter_type=filter_type,
        num_iterations=num_iterations,
        smooth_displacement_field=smooth_displacement_field,
        standard_deviations=standard_deviations,
        intensity_difference_threshold=intensity_difference_threshold,
        maximum_rms_error=maximum_rms_error,
    )

    # Convert input images to SimpleITK images
    reference_image = convert_to_sitk_image(reference_image)
    moving_image = convert_to_sitk_image(moving_image)

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because
    # of a constraint imposed by the Demons filters.

    df_size = reference_image.GetSize()
    df_spacing = reference_image.GetSpacing()

    initial_displacement_field = sitk.Image(df_size, sitk.sitkVectorFloat64, reference_image.GetDimension())
    initial_displacement_field.SetSpacing(df_spacing)
    initial_displacement_field.SetOrigin(reference_image.GetOrigin())

    # Run the registration.
    initial_displacement_field = sitk.Resample(initial_displacement_field, reference_image)
    initial_displacement_field = registration_algorithm.Execute(
        reference_image, moving_image, initial_displacement_field
    )
    displacement_field_transform = sitk.DisplacementFieldTransform(initial_displacement_field)

    # Extract the displacement field from the transform
    displacement_field = displacement_field_transform.GetDisplacementField()

    # Convert the displacement field to a SimpleITK image
    displacement_field_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(displacement_field), isVector=True)
    displacement_field_image.CopyInformation(reference_image)

    return convert_to_tensor(displacement_field_image).permute([2, 0, 1])


def multiscale_demons_displacement(
    reference_image: TensorOrNdarray,
    moving_image: TensorOrNdarray,
    filter_type: DemonsFilterType = DemonsFilterType.SYMMETRIC_FORCES,
    num_iterations: int = 100,
    smooth_displacement_field: bool = True,
    standard_deviations: float = 1.5,
    intensity_difference_threshold: Optional[float] = None,
    maximum_rms_error: Optional[float] = None,
) -> torch.Tensor:
    """Perform multiscale demons registration using SimpleITK.

    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.

    Parameters
    ----------
    reference_image : torch.Tensor
        A reference (grayscale) image of shape (H, W) or (D, H, W).
    moving_image : torch.Tensor
        A sequence of (grayscale) images (moving image) of shape (N, H, W) or (N, D, H, W) to register on
        the reference_image.
    filter_type : DemonsFilterType, optional
        Type of the Demons filter (DemonsFilterType.DEMONS, DemonsFilterType.FAST_SYMMETRIC_FORCES,
        DemonsFilterType.SYMMETRIC_FORCES, DemonsFilterType.DIFFEOMORPHIC). Default: DemonsFilterType.SYMMETRIC_FORCES.
    num_iterations : int
        Number of iterations for the Demons filter. Default: 100.
    smooth_displacement_field : bool
        Whether to smooth the displacement field. Default: True.
    standard_deviations : float
        Standard deviations for Gaussian smoothing. Default: 1.5.
    intensity_difference_threshold : float, optional
        Intensity difference threshold. Default: None.
    maximum_rms_error : float, optional
        Maximum RMS error. Default: None.

    Returns
    -------
    torch.Tensor
        Displacement field tensor of shape (N, 2, H, W) or (N, 3, D, H, W).
    """

    if (reference_image.ndim + 1) != (moving_image.ndim):
        raise ValueError(
            f"Expected the reference image to have one less dimension than the moving image, "
            f"corresponding to the number of frames in the sequence. Instead,"
            f"received shapes {reference_image.shape} and {moving_image.shape} for the "
            f"reference and moving images, respectively."
        )

    displacement = []

    for _, frame in enumerate(moving_image):

        flow = simpleitk_multiscale_demons_registration(
            reference_image=reference_image,
            moving_image=frame,
            filter_type=filter_type,
            num_iterations=num_iterations,
            smooth_displacement_field=smooth_displacement_field,
            standard_deviations=standard_deviations,
            intensity_difference_threshold=intensity_difference_threshold,
            maximum_rms_error=maximum_rms_error,
        )

        displacement.append(flow)

    return torch.stack(displacement, dim=0)
