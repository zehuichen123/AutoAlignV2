# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                          points_cam2img)
from ..builder import FUSION_LAYERS
from . import apply_3d_transformation
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
import math 

def point_sample(img_meta,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True,
                 img_id=0):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to camera coordinate
    pts_2d_with_depth = points_cam2img(points, proj_mat, with_depth=True)
    pts_depth = pts_2d_with_depth[:, -1]
    pts_2d = pts_2d_with_depth[:, :2]

    valid_depth_idx = (pts_depth > 0).reshape(-1,)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    # if img_flip:
    #     # by default we take it as horizontal flip
    #     # use img_shape before padding for flip
    #     orig_h, orig_w = img_shape
    #     coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1

    valid_y_idx = ((coor_y >= -1) & (coor_y <= 1)).reshape(-1,)
    valid_x_idx = ((coor_x >= -1) & (coor_y <= 1)).reshape(-1,)

    grid = torch.cat([coor_x, coor_y],
                     dim=1)

    valid_idx = valid_depth_idx & valid_y_idx & valid_x_idx
    return grid, valid_idx

@FUSION_LAYERS.register_module()
class MultiVoxelDeformFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Defaults to 'LIDAR'.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True,
                 num_heads=4,
                 num_points=8,
                 fix_offset=True,
                 im2col_step=64,
                 multi_input=''):
        super(MultiVoxelDeformFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.mid_channels = mid_channels
        self.num_points = num_points
        self.num_heads = num_heads
        self.num_levels = len(img_levels)
        self.fix_offset = fix_offset
        self.im2col_step = im2col_step
        self.multi_input = multi_input

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        # emprically find this projection provides no improvement
        # self.pts_proj = nn.Linear(pts_channels, mid_channels)
        # self.img_proj = nn.Linear(mid_channels if lateral_conv else img_channels[0], mid_channels)

        self.deform_sampling_offsets = nn.Linear(
                mid_channels, num_heads * self.num_levels * num_points * 2
            )
        self.attention_weights = nn.Linear(mid_channels,
                    num_heads * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(mid_channels, mid_channels)

        if self.fix_offset:
            self.deform_sampling_offsets.weight.requires_grad = False
            self.deform_sampling_offsets.bias.requires_grad = False


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Conv2d):
                normal_init(m, 0., std=1.0)
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.deform_sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.deform_sampling_offsets.bias.data = grid_init.view(-1)

    def forward(self, img_feats, voxel_coors, voxel_feats, \
                        img_metas, voxel_size, point_cloud_range):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features. NOTE this should be a list of list
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, voxel_coors, voxel_feats, \
                            img_metas, voxel_size, point_cloud_range)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(voxel_feats)

        # fuse_out = img_pre_fuse + pts_pre_fuse
        fuse_out = torch.cat([img_pre_fuse, pts_pre_fuse], dim=-1)
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats, voxel_coors, voxel_feats, img_metas, \
                        voxel_size, point_cloud_range):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        start_iter = 0
        # convert level-based multi-feature into batch based multi-feature
        num_camera = img_ins[0].shape[0] // len(img_metas)
        multi_feat_lists = []
        for i in range(len(img_metas)):
            feat_list = []
            for level in range(len(self.img_levels)):
                feat_list.append(img_ins[level][i * num_camera: (i + 1) * num_camera])
            multi_feat_lists.append(feat_list)

        for i in range(len(img_metas)):
            voxel_coors_per_img = voxel_coors[voxel_coors[:, 0] == i]
            x = (voxel_coors_per_img[:, 3] + 0.5) * voxel_size[0] + point_cloud_range[0]
            y = (voxel_coors_per_img[:, 2] + 0.5) * voxel_size[1] + point_cloud_range[1]
            z = (voxel_coors_per_img[:, 1] + 0.5) * voxel_size[2] + point_cloud_range[2]
            x = x.unsqueeze(-1); y = y.unsqueeze(-1); z = z.unsqueeze(-1)
            decoded_voxel_coors = torch.cat([x, y, z], dim=-1)
            num_voxels = decoded_voxel_coors.shape[0]
            voxel_feat = voxel_feats[start_iter: start_iter + num_voxels]
            img_feats_per_point.append(
                self.sample_single(multi_feat_lists[i], \
                decoded_voxel_coors, voxel_feat, img_metas[i]))
            start_iter += num_voxels
        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats, pts, pts_feats, img_meta):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (1, num_camera, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat_list = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        num_camera = img_feats[0].shape[0]
        # align pts_feats from each camera
        assign_mask = pts.new_zeros((pts.shape[0]), dtype=torch.bool)
        final_img_pts = pts.new_zeros((pts.shape[0], self.mid_channels))
        for camera_id in range(num_camera):
            grid, valid_idx = point_sample(
                img_meta=img_meta,
                points=pts,
                proj_mat=pts.new_tensor(proj_mat_list[camera_id]),
                coord_type=self.coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_meta['input_shape'][:2],
                img_shape=img_meta['img_shape'][:2],
                aligned=self.aligned,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                img_id=camera_id,
            )
            assign_idx = (~assign_mask) & valid_idx
            assign_mask |= assign_idx
            valid_grid = grid[assign_idx].unsqueeze(0).unsqueeze(0)

            # align_corner=True provides higher performance
            aligned = True
            mode = 'bilinear' if aligned else 'nearest'
            valid_ref_feats = 0.
            for ii in range(len(img_feats)):
                img_feat = F.grid_sample(
                    img_feats[ii][camera_id].unsqueeze(0),
                    valid_grid,
                    mode=mode,
                    padding_mode='zeros',
                    align_corners=False)  # 1xCx1xN feats
                valid_ref_feats += img_feat
            valid_ref_feats = valid_ref_feats.squeeze(-2).permute(0, 2, 1)

            valid_grid = valid_grid.permute(0, 2, 1, 3).repeat(1, 1, self.num_levels, 1)
            src_flattens = []; spatial_shapes = []
            lvl_pos_embed_flatten = []
            img_feats_per_camera = [img_feat[camera_id: camera_id+1] for img_feat in img_feats]
            for i in range(len(img_feats_per_camera)):
                bs, c, h, w = img_feats_per_camera[i].shape
                spatial_shapes.append((h, w))
                flatten_feat = img_feats_per_camera[i].view(bs, c, h, w).flatten(2).transpose(1, 2)
                src_flattens.append(flatten_feat)
            value_flatten = torch.cat(src_flattens, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=flatten_feat.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_pts_feats = pts_feats[assign_idx].unsqueeze(0)

            # valid_ref_feats = self.img_proj(valid_ref_feats)
            # valid_pts_feats = self.pts_proj(valid_pts_feats)

            if self.multi_input == 'concat':
                query_feat = torch.cat([valid_ref_feats, valid_pts_feats], dim=-1)
            elif self.multi_input == 'multiply':
                query_feat = valid_ref_feats * valid_pts_feats
            elif self.multi_input == 'multiply_pts_detach':
                query_feat = valid_ref_feats * valid_pts_feats.detach()
            elif self.multi_input == 'pts':
                query_feat = valid_pts_feats
            elif self.multi_input == 'pts_detach':
                query_feat = valid_pts_feats.detach()
            elif self.multi_input == 'img':
                query_feat = valid_ref_feats
            elif self.multi_input == 'img_detach':
                query_feat = valid_ref_feats.detach()
            else:
                raise Exception

            num_query = query_feat.shape[1]
            sampling_offsets = self.deform_sampling_offsets(query_feat).view(
                1, num_query, self.num_heads, self.num_levels, self.num_points, 2
            )
            attention_weights = self.attention_weights(query_feat).view(
                1, num_query, self.num_heads, self.num_levels * self.num_points
            )
            attention_weights = attention_weights.softmax(-1)

            value_flatten = self.value_proj(value_flatten)
            _, num_value, _ = value_flatten.shape
            value_flatten = value_flatten.view(1, num_value, self.num_heads, -1)

            if valid_grid.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = valid_grid[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            elif valid_grid.shape[-1] == 4:
                sampling_locations = valid_grid[:, :, None, :, None, :2] \
                    + sampling_offsets / self.num_points \
                    * reference_points_cam[:, :, None, :, None, 2:] \
                    * 0.5
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2 or 4, but get {reference_points_cam.shape[-1]} instead.')
                    
            if torch.cuda.is_available() and value_flatten.is_cuda:
                output = MultiScaleDeformableAttnFunction.apply(
                    value_flatten, spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
            else:
                # WON'T REACH HERE
                print("Won't Reach Here")
                output = multi_scale_deformable_attn_pytorch(
                    value_flatten, spatial_shapes, sampling_locations, attention_weights)
            output = output + valid_ref_feats
            final_img_pts[assign_idx] = output.squeeze(0)
        return final_img_pts
