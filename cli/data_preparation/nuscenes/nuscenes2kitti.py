import os

import numpy as np
import numpy.linalg as la
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

import nuscenes.utils.geometry_utils as geoutils
from nuscenes import NuScenes
from nuscenes.utils import splits

import click


def convert_scenes(dataroot, output_dir, normalize_remission, save_images, mini, trainval=True):
    if mini:
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
        # Get sequence names
        scenes = splits.mini_train + splits.mini_val
    elif trainval:
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        # Get sequence names
        scenes = splits.train + splits.val
    else:
        nusc = NuScenes(version='v1.0-test', dataroot=dataroot, verbose=True)
        scenes = splits.test

    # Create sequences dirs
    for scene in scenes:
        dirname = scene[6:]
        output_seq_dir = os.path.join(output_dir, dirname)
        if not os.path.exists(output_seq_dir):
            os.makedirs(output_seq_dir, exist_ok=True)

    # Iterate over all scenes (sequences)
    for scene in tqdm(nusc.scene):
        # Create directories: sequence, velodyne, labels
        dirname = scene['name'][6:]
        output_seq_dir = os.path.join(output_dir, dirname)
        pose_f = os.path.join(output_seq_dir, 'poses.txt')
        calib_f = os.path.join(output_seq_dir, 'calib.txt')
        vel_dir = os.path.join(output_seq_dir, 'velodyne')
        lab_dir = os.path.join(output_seq_dir, 'labels')
        if not os.path.exists(output_seq_dir):
            os.makedirs(output_seq_dir, exist_ok=True)
        if not os.path.exists(vel_dir):
            os.makedirs(vel_dir, exist_ok=True)
        if not os.path.exists(lab_dir):
            os.makedirs(lab_dir, exist_ok=True)

        # Create dummy calib file
        calib_file = open(calib_f, "w")
        calib_file.write("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.write("P1: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.write("P2: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.write("P3: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.write("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.close()

        next_sample = scene['first_sample_token']

        lidar_filenames = []
        poses = []
        files_mapping = []
        lidar_tokens = []

        # Iterate over all samples (scans) in the sequence
        while next_sample != '':
            # Current sample data
            sample = nusc.get('sample', next_sample)
            # Get token for the next sample
            next_sample = sample['next']

            # Get lidar, semantic and panoptic filenames
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            scan = np.fromfile(os.path.join(dataroot, lidar_data["filename"]), dtype=np.float32)
            # Save scan
            points = scan.reshape((-1, 5))[:, :4]
            if normalize_remission:
                # ensure that remission is in [0,1]
                max_remission = np.max(points[:, 3])
                min_remission = np.min(points[:, 3])
                points[:, 3] = (points[:, 3] - min_remission) / (max_remission - min_remission)
            # velodyne bin file
            output_filename = os.path.join(vel_dir, "{:06d}.bin".format(len(lidar_filenames)))
            points.tofile(output_filename)
            # Compute pose
            calib_data = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            egopose_data = nusc.get('ego_pose', lidar_data["ego_pose_token"])
            car_to_velo = geoutils.transform_matrix(calib_data["translation"], Quaternion(calib_data['rotation']))
            pose_car = geoutils.transform_matrix(egopose_data["translation"], Quaternion(egopose_data['rotation']))
            pose = np.dot(pose_car, car_to_velo)
            poses.append(pose)

            # Compute labels for train and val scenes
            if trainval:
                sem_lab_f = nusc.get('lidarseg', lidar_token)['filename']
                sem_lab = np.fromfile(os.path.join(dataroot, sem_lab_f), dtype=np.uint8)
                pan_lab_f = nusc.get('panoptic', lidar_token)['filename']
                pan_lab = np.load(os.path.join(dataroot, pan_lab_f))['data']
                # sem labels from panoptic labels
                # sem_lab2 = (pan_lab // 1000).astype(np.uint8)
                # ins labels from panoptic labels
                ins_lab = pan_lab % 1000
                # Kitti style panoptic labels
                panoptic_labels = sem_lab.reshape(-1, 1) \
                    + ((ins_lab.astype(np.uint32) << 16) & 0xFFFF0000).reshape(-1, 1)

                # Save labels
                lab_output_filename = os.path.join(lab_dir, "{:06d}.label".format(len(lidar_filenames)))
                panoptic_labels.tofile(lab_output_filename)

            # Keep list of filenames and tokens
            files_mapping.append(lidar_data["filename"])
            lidar_filenames.append(os.path.join(dataroot, lidar_data["filename"]))
            lidar_tokens.append(lidar_token)

        # Create pose file
        ref = la.inv(poses[0])
        pose_file = open(pose_f, "w")
        for pose in poses:
            pose_str = [str(v) for v in (np.dot(ref, pose))[:3, :4].flatten()]
            pose_file.write(" ".join(pose_str))
            pose_file.write("\n")
        pose_file.close()

        # Save filenames and tokens for each point cloud
        files_mapping_f = os.path.join(output_seq_dir, 'files_mapping.txt')
        files_mapping_file = open(files_mapping_f, "w")
        for f in files_mapping:
            files_mapping_file.write(os.path.join(dataroot, f))
            files_mapping_file.write("\n")
        files_mapping_file.close()

        lidar_tokens_f = os.path.join(output_seq_dir, 'lidar_tokens.txt')
        lidar_tokens_file = open(lidar_tokens_f, "w")
        for token in lidar_tokens:
            lidar_tokens_file.write(token)
            lidar_tokens_file.write("\n")
        lidar_tokens_file.close()

        if save_images:
            image_dir = os.path.join(output_seq_dir, "image_2/")
            if not os.path.exists(image_dir):
                os.makedirs(image_dir, exist_ok=True)

            next_image = nusc.get('sample', scene["first_sample_token"])["data"]["CAM_FRONT"]
            original = []
            image_filenames = []
            # todo: get relative pose to velodyne.
            # poses = []
            while next_image != "":
                image_data = nusc.get('sample_data', next_image)
                output_filename = os.path.join(image_dir, "{:06d}.png".format(len(image_filenames)))
                image = Image.open(os.path.join(dataroot, image_data["filename"]))  # open jpg.
                image.save(output_filename)  # and save as png.
                original.append(("{:05d}.png".format(len(image_filenames)), image_data["filename"]))
                image_filenames.append(os.path.join(dataroot, image_data["filename"]))
                next_image = image_data["next"]

            original_file = open(os.path.join(output_seq_dir, "original_images_2.txt"), "w")
            for pair in original:
                original_file.write(pair[0] + ":" + pair[1] + "\n")
            original_file.close()


@click.command()
@click.option('--nuscenes_dir', type=str, default=None, required=True, help='dataroot directory of nuscenes dataset')
@click.option('--output_dir', type=str, default=None, required=True, help='directory where to save the sequences')
@click.option('--normalize_remission', is_flag=True, help='normalize remission values in range [0,1]')
@click.option('--mini', is_flag=True, help='convert only mini set')
@click.option('--save_images', is_flag=True, help='save front camera images')
def main(nuscenes_dir, output_dir, normalize_remission, mini, save_images):
    if mini:
        convert_scenes(nuscenes_dir, output_dir, normalize_remission, save_images, mini)
    else:
        convert_scenes(nuscenes_dir, output_dir, normalize_remission, save_images, mini)
        convert_scenes(nuscenes_dir, output_dir, normalize_remission, save_images, mini, trainval=False)


if __name__ == "__main__":
    main()
