import inspect
import pprint
import numpy as np
import io
import pickle
from efg.data.voxel_generator import VoxelGenerator
from efg.data.registry import PROCESSORS
from efg.data.augmentations3d import Augmentation

@PROCESSORS.register()
class ClipVoxelization(Augmentation):
    def __init__(self, pc_range, voxel_size, max_points_in_voxel, max_voxel_num, num_clips=None):
        super().__init__()
        self._init(locals())
        self.num_clips = num_clips
        self.voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=max_points_in_voxel,
            max_voxels=max_voxel_num,
        )

    def __call__(self, points, info):
        # [0, -40, -3, 70.4, 40, 1]
        # pc_range = self.voxel_generator.point_cloud_range
        # grid_size = self.voxel_generator.grid_size
        # # [352, 400]

        # # import pdb;pdb.set_trace()
        # point_voxels_list = []
        # for idx in range(1,self.num_clips):
        #     time_mask1 = np.abs((points[:,-1] - 0.1*(idx-1))) <  1e-2,
        #     time_mask2 = np.abs((points[:,-1] - 0.1*idx)) < 1e-2
        #     point_clip1 = points[time_mask1]
        #     point_clip1[:,-1] = 0.0
        #     point_clip2 = points[time_mask2]
        #     point_clip2[:,-1] = 0.1
        #     point_clip = np.concatenate([point_clip1,point_clip2],0)
        #     voxels, coordinates, num_points_per_voxel = self.voxel_generator.generate(point_clip)
        #     num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        #     point_voxels = dict(
        #         voxels=voxels,
        #         points=point_clip,
        #         coordinates=coordinates,
        #         num_points_per_voxel=num_points_per_voxel,
        #         num_voxels=num_voxels,
        #         shape=grid_size,
        #         range=pc_range,
        #         size=self.voxel_size,
        #     )
        #     point_voxels_list.insert(0,point_voxels)
        return [{'points': points}], info