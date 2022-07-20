# Copyright (c) OpenMMLab. All rights reserved.
from .coord_transform import (apply_3d_transformation, bbox_2d_transform,
                              coord_2d_transform)
from .point_fusion import PointFusion
from .vote_fusion import VoteFusion
from .multi_point_fusion import MultiPointFusion
# from .deform_point_fusion import DeformPointFusion
# from .deform_point_fusion_v2 import DeformPointFusionV2
from .multi_voxel_fusion import MultiVoxelFusion
# from .multi_voxel_deform_fusion import MultiVoxelDeformFusion
# from .multi_voxel_deform_fusion_v2 import MultiVoxelDeformFusionV2
# from .multi_voxel_deform_fusion_v2_debug import MultiVoxelDeformFusionV2Debug
# from .multi_voxel_deform_fusion_v3 import MultiVoxelDeformFusionV3
from .multi_voxel_fusion_fast import MultiVoxelFusionFast
from .multi_voxel_fusion_fast_add import MultiVoxelFusionFastAdd
__all__ = [
    'PointFusion', 'VoteFusion', 'apply_3d_transformation',
    'bbox_2d_transform', 'coord_2d_transform', 'MultiPointFusion',
    # 'DeformPointFusion', 
    'MultiVoxelFusion', 
    # 'MultiVoxelDeformFusion',
    # 'DeformPointFusionV2', 'MultiVoxelDeformFusionV2Debug', 'MultiVoxelDeformFusionV3',
    'MultiVoxelFusionFast', 'MultiVoxelFusionFastAdd'
]
