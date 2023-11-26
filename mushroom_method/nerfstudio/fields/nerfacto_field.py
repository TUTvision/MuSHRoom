# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.encodings import NeRFEncoding

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
import json

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_directions(directions: TensorType["bs":..., 3]):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class TCNNNerfactoField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.weight_norm = True

        base_res = 16
        features_per_level = 2
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
                "interpolation": "Smoothstep",
            },
        )
        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=6,
            min_freq_exp=0.0,
            max_freq_exp=6 - 1,
            include_input=False,
        )

        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.transient_embedding_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_transient,
                    "n_hidden_layers": num_layers_transient - 1,
                },
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.n_output_dims)
            self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.n_output_dims)
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.n_output_dims)

        # semantics
        if self.use_semantics:
            self.mlp_semantics = tcnn.Network(
                n_input_dims=self.geo_feat_dim,
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.n_output_dims, num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = tcnn.Network(
                n_input_dims=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                n_output_dims=hidden_dim_transient,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.n_output_dims)

        dims = [hidden_dim for _ in range(num_layers)]
        in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims
        dims = [in_dim] + dims + [1 + self.geo_feat_dim]
        self.num_layers = len(dims)
        # TODO check how to merge skip_in to config
        self.skip_in = [4]

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            torch.nn.init.kaiming_uniform_(lin.weight.data)
            torch.nn.init.zeros_(lin.bias.data)

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)
            # print("=======", lin.weight.shape)
            setattr(self, "dlin" + str(l), lin)

        dims = [hidden_dim_color for _ in range(num_layers_color)]

        # point, view_direction, normal, feature, embedding
        in_dim = 3 + self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.embedding_appearance.get_out_dim()

        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for l in range(0, self.num_layers_color - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            torch.nn.init.kaiming_uniform_(lin.weight.data)
            torch.nn.init.zeros_(lin.bias.data)

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)
            # print("=======", lin.weight.shape)
            setattr(self, "clin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = nn.ReLU()
        self.laplace_density = LaplaceDensity(init_val=0.7)

        # initialzie R and T

    def forward_densitynetwork(self, inputs):
        """forward the geonetwork"""

        positions = (inputs + 2.0) / 4.0  # inputs [98304, 3]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        feature = self.encoding(positions)  # [98304, 32]

        pe = self.position_encoding(inputs)  # sinousoidal encoding [98304, 36]

        inputs = torch.cat((inputs, pe, feature), dim=-1)  # [98304, 71]

        x = inputs  # 98304,328

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "dlin" + str(l))  # linear (71, 256), (256, 256), (256, 257)

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)  # 256, 257

            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x  # [98304, 257]

    def get_color(self, points, directions, density_embedding, embedded_appearance):

        h = [
            points,
            directions,  # 16
            density_embedding,
            embedded_appearance.view(-1, self.appearance_embedding_dim),  # appearance embedding
        ]

        h = torch.cat(h, dim=-1)  # [98304, 321]

        for l in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(l))

            h = lin(h)

            if l < self.num_layers_color - 2:
                h = self.relu(h)
        rgb = self.sigmoid(h)
        return rgb

    def get_normals(self) -> TensorType[..., 3]:
        """Computes and returns a tensor of normals.

        Args:
            density: Tensor of densities.
        """
        assert self._sample_locations is not None, "Sample locations must be set before calling get_normals."
        assert self._density_before_activation is not None, "Density must be set before calling get_normals."
        assert (
            self._sample_locations.shape[:-1] == self._density_before_activation.shape[:-1]
        ), "Sample locations and density must have the same shape besides the last dimension."

        self._density_before_activation.backward(
            gradient=torch.ones_like(self._density_before_activation), inputs=self._sample_locations, retain_graph=True
        )
        normals = -torch.nn.functional.normalize(self._sample_locations.grad, dim=-1)
        return normals

    def get_outputs(self, ray_samples: RaySamples, compute_normals: bool = True):
        # assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs_shape = ray_samples.frustums.directions.shape[:-1]  # [n_rays, n_sample]

        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)

        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)  # [n_rays * n_samples, 16]

        points = ray_samples.frustums.get_positions()
        points = self.spatial_distortion(points)
        positions_flat = points.view(-1, 3)  # point cloud点 归一化后，[n,3]

        if compute_normals:
            with torch.enable_grad():

                density = self.forward_densitynetwork(positions_flat)
                density_before_activation, density_embedding = torch.split(density, [1, self.geo_feat_dim], dim=-1)

                self._density_before_activation = density_before_activation
        else:
            density = self.forward_densitynetwork(positions_flat)
            density_before_activation, density_embedding = torch.split(density, [1, self.geo_feat_dim], dim=-1)

            self._density_before_activation = density_before_activation

        density = trunc_exp(density_before_activation.to(points)).view(*outputs_shape, -1)
        density = density.view(*outputs_shape, -1)
        outputs.update({FieldHeadNames.DENSITY: density})

        # # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat_norm = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat_norm, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)  # [4096, 48, 3]

        rgb = self.get_color(positions_flat, d, density_embedding, embedded_appearance)
        rgb = rgb.view(*outputs_shape, -1)  # [98304, 3]

        outputs.update({FieldHeadNames.RGB: rgb})

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals().view(*outputs_shape, -1)
                outputs.update({FieldHeadNames.NORMALS: normals})

        return outputs
