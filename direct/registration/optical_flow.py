# Copyright (c) DIRECT Contributors

"""Optical flow estimation functions for image registration using scikit-image."""

import numpy as np
import torch
from skimage.registration import optical_flow_ilk, optical_flow_tvl1

from direct.types import DirectEnum


def skimage_optical_flow_ilk(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    radius: int = 7,
    num_warp: int = 10,
    gaussian: bool = False,
    prefilter: bool = False,
) -> np.ndarray:
    """Optical flow estimator using the iterative Lucas-Kanade (iLK) solver [1]_.

    iLK is applied at each level of the image pyramid and is a fast and robust alternative to TVL1 algorithm although
    less accurate for rendering flat surfaces and object boundaries (see [2]_).

    For more information, see the scikit-image documentation at [3]_.

    Parameters
    ----------
    reference_image : ndarray
        A grayscale image of the sequence of shape ([D], H, W).
    moving_image : ndarray
        A grayscale image of the sequence of shape ([D], H, W). This image is warped to match the reference_image.
    radius : int, optional
        Radius of the window considered around each pixel. Defaut: 7.
    num_warp : int, optional
        Number of times moving_image is warped. Default: 10.
    gaussian : bool, optional
        If True, a Gaussian kernel is used for the local integration.
        Otherwise, a uniform kernel is used. Default: False.
    prefilter : bool, optional
        Whether to prefilter the estimated optical flow before each image warp.
        When True, a median filter with window size 3 along each axis is applied.
        This helps to remove potential outliers. Default: False.

    Returns
    -------
    flow : ndarray
        The estimated optical flow components for each axis. Optical flow is of
        shape (reference_image.ndim, reference_image.shape)

    References
    ----------
    .. [1] Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique
        with an application to stereo vision. In Proceedings of the 7th International
        Joint Conference on Artificial Intelligence (IJCAI'81) (pp. 674-679).
    .. [2] Beauchemin, S. S., & Barron, J. L. (1995). The computation of optical flow.
        ACM Computing Surveys (CSUR), 27(3), 433-466.
    .. [3] https://scikit-image.org/docs/stable/api/skimage.registration.html#module-skimage.registration
    """
    return optical_flow_ilk(
        reference_image, moving_image, radius=radius, num_warp=num_warp, gaussian=gaussian, prefilter=prefilter
    )


def skimage_optical_flow_tvl1(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    attachment: float = 15,
    tightness: float = 0.3,
    num_warp: int = 5,
    num_iter: int = 10,
    tol: float = 1e-3,
    prefilter: bool = False,
):
    r"""Optical flow estimator using the TV-L1 solver ([1]_, [2]_, [3]_).

    The TV-L1 solver is applied at each level of the image pyramid.

    Parameters
    ----------
    reference_image : ndarray
        A grayscale image of the sequence of shape ([D], H, W).
    moving_image : ndarray
        A grayscale image of the sequence of shape ([D], H, W). This image is warped to match the reference_image.
    attachment : float, optional
        Attachment parameter (:math:`\lambda` in [1]_). The smaller this parameter is, the smoother
        the returned result will be. Default: 15.
    tightness : float, optional
        Tightness parameter (:math:`\theta` in [1]_). It should have a small value in order to maintain
        attachment and regularization parts in correspondence. Default: 0.3.
    num_warp : int, optional
        Number of times moving_image is warped. Default: 5.
    num_iter : int, optional
        Number of fixed point iterations. Default: 10.
    tol : float, optional
        Tolerance used as stopping criterion based on the L2 distance between two
        consecutive values of (u, v). Default: 1e-3.
    prefilter : bool, optional
        Whether to prefilter the estimated optical flow before each image warp. When True,
        a median filter with window size 3 along each axis is applied. This helps to
        remove potential outliers. Default: False.

    Returns
    -------
    flow : ndarray
        The estimated optical flow components for each axis. Optical flow is of
        shape (reference_image.ndim, reference_image.shape)

    References
    ----------
    .. [1] Zach, C., Pock, T., & Bischof, H. (2007). A duality based approach for real-time TV-L1 optical flow.
        In Joint Pattern Recognition Symposium (pp. 214-223). Springer, Berlin, Heidelberg.
    .. [2] Wedel, A., Pock, T., Zach, C., Bischof, H., & Cremers, D. (2009). An improved algorithm for TV-L 1 optical
        flow. In Statistical and geometrical approaches to visual motion analysis (pp. 23-45).
        Springer, Berlin, Heidelberg.
    .. [3] Pérez, J. S., Meinhardt-Llopis, E., & Facciolo, G. (2013). TV-L1 optical flow estimation.
        Image Processing On Line, 2013, 137-150.
    .. [4] https://scikit-image.org/docs/stable/api/skimage.registration.html#module-skimage.registration
    """

    return optical_flow_tvl1(
        reference_image,
        moving_image,
        attachment=attachment,
        tightness=tightness,
        num_warp=num_warp,
        num_iter=num_iter,
        tol=tol,
        prefilter=prefilter,
    )


class OpticalFlowEstimatorType(DirectEnum):
    ILK = "iterative_lucas_kanade"
    TV_L1 = "tv_l1"


def optical_flow_displacement(
    reference_image: torch.Tensor,
    moving_image: torch.Tensor,
    estimator_type: OpticalFlowEstimatorType = OpticalFlowEstimatorType.TV_L1,
    **kwargs,
) -> torch.Tensor:
    """Estimate the optical flow between a reference and a sequence of images (moving image).

    Parameters
    ----------
    reference_image : torch.Tensor
        A reference (grayscale) image of shape (H, W) or (D, H, W).
    moving_image : torch.Tensor
        A sequence of (grayscale) images (moving image) of shape (N, H, W) or (N, D, H, W) to register on
        the reference_image.
    estimator_type : OpticalFlowEstimatorType
        The type of optical flow estimator to use.
    **kwargs : dict
        Additional keyword arguments to pass to the optical flow estimator. See the documentation for the specific
        estimator for more information.

    Returns
    -------
    displacement : torch.Tensor
        The estimated optical flow tensor of shape (N, 2, H, W, 2) or (N, 3, D, H, W).
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
        if estimator_type == OpticalFlowEstimatorType.ILK:
            flow = skimage_optical_flow_ilk(reference_image.numpy(), frame.numpy(), **kwargs)
        else:
            flow = skimage_optical_flow_tvl1(reference_image.numpy(), frame.numpy(), **kwargs)
        flow = np.stack([flow[1], flow[0]], axis=0)  # Necessary for custom warp function
        displacement.append(torch.from_numpy(flow))

    return torch.stack(displacement, dim=0)
