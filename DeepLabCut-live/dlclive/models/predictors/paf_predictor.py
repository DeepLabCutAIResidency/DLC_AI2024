#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dlclive.models.predictors.base import PREDICTORS, BasePredictor

Graph = list[tuple[int, int]]


@PREDICTORS.register_module
class PartAffinityFieldPredictor(BasePredictor):
    """Predictor class for multiple animal pose estimation with part affinity fields.

    Args:
        num_animals: Number of animals in the project.
        num_multibodyparts: Number of animal's body parts (ignoring unique body parts).
        graph: Part affinity field graph edges.
        edges_to_keep: List of indices in `graph` of the edges to keep.
        locref_stdev: Standard deviation for location refinement.
        nms_radius: Radius of the Gaussian kernel.
        sigma: Width of the 2D Gaussian distribution.

    Returns:
        Regressed keypoints from heatmaps, locref_maps and part affinity fields, as in Tensorflow maDLC.
    """

    default_init = {
        "locref_stdev": 7.2801,
        "nms_radius": 5,
        "sigma": 1,
        "min_affinity": 0.05,
    }

    def __init__(
        self,
        num_animals: int,
        num_multibodyparts: int,
        num_uniquebodyparts: int,
        graph: Graph,
        edges_to_keep: list[int],
        locref_stdev: float,
        nms_radius: int,
        sigma: float,
        apply_sigmoid: bool = True,
        clip_scores: bool = False,
        return_preds: bool = False,
    ):
        """Initialize the PartAffinityFieldPredictor class.

        Args:
            num_animals: Number of animals in the project.
            num_multibodyparts: Number of animal's body parts (ignoring unique body parts).
            num_uniquebodyparts: Number of unique body parts.
            graph: Part affinity field graph edges.
            edges_to_keep: List of indices in `graph` of the edges to keep.
            locref_stdev: Standard deviation for location refinement.
            nms_radius: Radius of the Gaussian kernel.
            sigma: Width of the 2D Gaussian distribution.
            return_preds: Whether to return predictions alongside the animals' poses

        Returns:
            None
        """
        super().__init__()
        self.num_animals = num_animals
        self.num_multibodyparts = num_multibodyparts
        self.num_uniquebodyparts = num_uniquebodyparts
        self.graph = graph
        self.edges_to_keep = edges_to_keep
        self.locref_stdev = locref_stdev
        self.nms_radius = nms_radius
        self.return_preds = return_preds
        self.sigma = sigma
        self.apply_sigmoid = apply_sigmoid
        self.clip_scores = clip_scores
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self, stride: float, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass of PartAffinityFieldPredictor. Gets predictions from model output.

        Args:
            stride: the stride of the model
            outputs: Output tensors from previous layers.
                output = heatmaps, locref, pafs
                heatmaps: torch.Tensor([batch_size, num_joints, height, width])
                locref: torch.Tensor([batch_size, num_joints, height, width])

        Returns:
            A dictionary containing a "poses" key with the output tensor as value.

        Example:
            >>> predictor = PartAffinityFieldPredictor(num_animals=3, location_refinement=True, locref_stdev=7.2801)
            >>> output = (torch.rand(32, 17, 64, 64), torch.rand(32, 34, 64, 64), torch.rand(32, 136, 64, 64))
            >>> stride = 8
            >>> poses = predictor.forward(stride, output)
        """
        heatmaps = outputs["heatmap"]
        locrefs = outputs["locref"]
        pafs = outputs["paf"]
        scale_factors = stride, stride
        batch_size, n_channels, height, width = heatmaps.shape

        if self.apply_sigmoid:
            heatmaps = self.sigmoid(heatmaps)

        # Filter predicted heatmaps with a 2D Gaussian kernel as in:
        # https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.pdf
        kernel = self.make_2d_gaussian_kernel(
            sigma=self.sigma, size=self.nms_radius * 2 + 1
        )[None, None]
        kernel = kernel.repeat(n_channels, 1, 1, 1).to(heatmaps.device)
        heatmaps = F.conv2d(
            heatmaps, kernel, stride=1, padding="same", groups=n_channels
        )

        peaks = self.find_local_peak_indices_maxpool_nms(
            heatmaps, self.nms_radius, threshold=0.01
        )
        if ~torch.any(peaks):
            return {
                "poses": -torch.ones(
                    (batch_size, self.num_animals, self.num_multibodyparts, 5)
                )
            }

        locrefs = locrefs.reshape(batch_size, n_channels, 2, height, width)
        locrefs = locrefs * self.locref_stdev

        poses = -torch.ones((batch_size, self.num_animals, self.num_multibodyparts, 5))
        if self.clip_scores:
            poses[..., 2] = torch.clip(poses[..., 2], min=0, max=1)

        # TODO: FIXME
        return {"poses": poses}

    @staticmethod
    def find_local_peak_indices_maxpool_nms(
        input_: torch.Tensor, radius: int, threshold: float
    ) -> torch.Tensor:
        pooled = F.max_pool2d(input_, kernel_size=radius, stride=1, padding=radius // 2)
        maxima = input_ * torch.eq(input_, pooled).float()
        peak_indices = torch.nonzero(maxima >= threshold, as_tuple=False)
        return peak_indices.int()

    @staticmethod
    def make_2d_gaussian_kernel(sigma: float, size: int) -> torch.Tensor:
        k = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32) ** 2
        k = F.softmax(-k / (2 * (sigma**2)), dim=0)
        return torch.einsum("i,j->ij", k, k)

    @staticmethod
    def calc_peak_locations(
        locrefs: torch.Tensor,
        peak_inds_in_batch: torch.Tensor,
        strides: tuple[float, float],
        n_decimals: int = 3,
    ) -> torch.Tensor:
        s, b, r, c = peak_inds_in_batch.T
        stride_y, stride_x = strides
        strides = torch.Tensor((stride_x, stride_y)).to(locrefs.device)
        off = locrefs[s, b, :, r, c]
        loc = strides * peak_inds_in_batch[:, [3, 2]] + strides // 2 + off
        return torch.round(loc, decimals=n_decimals)
