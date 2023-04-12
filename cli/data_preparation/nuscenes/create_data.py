import argparse
import os
import pickle
from functools import reduce

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, transform_matrix

from efg.data.datasets.nuscenes.nusc_common import general_to_detection, read_file, read_sweep
from efg.geometry import box_ops


def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not os.path.exists(lidar_path):
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec["next"] == "":
                sd_rec = nusc.get("sample_data", sd_rec["next"])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def get_sample_data(
    nusc,
    sample_data_token,
    box_vis_level=BoxVisibility.ANY,
    selected_anntokens=None,
    use_flat_vehicle_coordinates=False,
):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        imsize = (sd_record["width"], sd_record["height"])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        # obtain box velocity in global coordinates
        box.velocity = nusc.box_velocity(box.token)
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)

        if sensor_record["modality"] == "camera" and not box_in_image(
            box, cam_intrinsic, imsize, vis_level=box_vis_level
        ):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, nsweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []

    ref_chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.
    chan = "LIDAR_TOP"  # The current channel.

    for sample in tqdm(nusc.sample):
        """Manual save info["sweeps"]"""
        # Get reference pose and timestamp
        # ref_chan == "LIDAR_TOP"
        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample["data"]["CAM_FRONT"]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True)
        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True
        )

        info = {
            "lidar_path": ref_lidar_path,
            "cam_front_path": ref_cam_path,
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }

        sample_data_token = sample["data"][chan]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        sweeps = []
        while len(sweeps) < nsweeps - 1:
            if curr_sd_rec["prev"] == "":
                if len(sweeps) == 0:
                    sweep = {
                        "lidar_path": ref_lidar_path,
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec["timestamp"] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                # Get past pose
                current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
                global_from_car = transform_matrix(
                    current_pose_rec["translation"], Quaternion(current_pose_rec["rotation"]), inverse=False
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get("calibrated_sensor", curr_sd_rec["calibrated_sensor_token"])
                car_from_current = transform_matrix(
                    current_cs_rec["translation"], Quaternion(current_cs_rec["rotation"]), inverse=False
                )

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])

                time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": curr_sd_rec["token"],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        assert (
            len(info["sweeps"]) == nsweeps - 1
        ), f"sweep {curr_sd_rec['token']} has {len(info['sweeps'])} sweeps, please repeat to num {nsweeps-1}."

        if not test:
            annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]

            # convert from nuScenes Lidar to waymo Lidar
            rot = Quaternion(axis=[0, 0, 1], degrees=-90)
            [rb.rotate(rot) for rb in ref_boxes]

            # raw annotations in nuScenes LIDAR_TOP coordinates
            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)  # x, y, z
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # w, l, h
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)  # vx, vy, vz
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)  # yaw
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.nan_to_num(np.concatenate([locs, dims, velocity[:, :2], rots], axis=1))

            mask = np.array(
                [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations],
                dtype=bool,
            ).reshape(-1)

            info["gt_boxes"] = gt_boxes[mask].astype(np.float32)
            info["gt_names"] = np.array([general_to_detection[name] for name in names])[mask]
            info["gt_boxes_token"] = tokens[mask]

            assert len(annotations) == len(gt_boxes) == len(velocity)

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def create_nuscenes_infos(root_path, version="v1.0-trainval", nsweeps=10):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    assert version in ["v1.0-trainval", "v1.0-test", "v1.0-mini"]

    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    test = "test" in version
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]

    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]["token"] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes])

    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(nusc, train_scenes, val_scenes, test, nsweeps=nsweeps)

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(os.path.join(root_path, f"infos_test_{nsweeps:02d}sweeps_withvelo_new.pkl"), "wb") as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}")
        with open(os.path.join(root_path, f"infos_train_{nsweeps:02d}sweeps_withvelo_new.pkl"), "wb") as f:
            pickle.dump(train_nusc_infos, f)
        with open(os.path.join(root_path, f"infos_val_{nsweeps:02d}sweeps_withvelo_new.pkl"), "wb") as f:
            pickle.dump(val_nusc_infos, f)


def create_groundtruth_database(
    data_path,
    info_path=None,
    used_classes=(
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ),
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    nsweeps=1,
):
    root_path = data_path

    if db_path is None:
        db_path = os.path.join(root_path, f"gt_database_train_{nsweeps:02d}sweeps_withvelo_new")
    if dbinfo_path is None:
        dbinfo_path = os.path.join(root_path, f"gt_database_train_{nsweeps:02d}sweeps_withvelo_new_infos.pkl")
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    # nuscenes dataset setting
    point_features = 5

    all_db_infos = {}
    group_counter = 0

    with open(info_path, "rb") as f:
        dataset_infos_all = pickle.load(f)

    for info in tqdm(dataset_infos_all):
        lidar_path = info["lidar_path"]

        points = read_file(str(lidar_path))

        # points[:, 3] /= 255
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (nsweeps - 1) <= len(info["sweeps"]), "nsweeps {} should not greater than list length {}.".format(
            nsweeps, len(info["sweeps"])
        )

        for i in range(nsweeps - 1):
            points_sweep, times_sweep = read_sweep(info["sweeps"][i])
            if points_sweep is None or times_sweep is None:
                continue
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.hstack([points, times])
        points[:, :2] = points[:, [1, 0]]
        points[:, 1] *= -1

        annos = {
            "boxes": info["gt_boxes"],
            "names": info["gt_names"],
            "tokens": info["gt_boxes_token"],
        }

        gt_boxes = annos["boxes"]
        names = annos["names"]

        num_obj = gt_boxes.shape[0]

        if num_obj == 0:
            continue

        group_dict = {}
        # group_ids = np.full([num_obj], -1, dtype=np.int64)

        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(num_obj, dtype=np.int64)

        difficulty = np.zeros(num_obj, dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        point_indices = box_ops.points_in_rbbox(points, gt_boxes)

        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]

                filename = f"{info['token']}_{names[i]}_{i}.bin"

                dirpath = os.path.join(db_path, names[i])
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                filepath = os.path.join(db_path, names[i], filename)
                with open(filepath, "w") as f:
                    try:
                        gt_points[:, :point_features].tofile(f)
                    except:
                        print("process {} files".format(info["token"]))
                        break

                if relative_path:
                    db_dump_path = os.path.join(
                        f"gt_database_train_{nsweeps:02d}sweeps_withvelo_new", names[i], filename
                    )
                else:
                    db_dump_path = filepath

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "token": info["token"],
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }

                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1

                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    print("dataset length: ", len(dataset_infos_all))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("nuScenes dataset preparation")
    parser.add_argument("--root-path", default=None)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--nsweeps", default=1, type=int)

    args = parser.parse_args()

    create_nuscenes_infos(args.root_path, args.version, args.nsweeps)

    info_path = os.path.join(args.root_path, f"infos_train_{args.nsweeps:02d}sweeps_withvelo_new.pkl")

    if args.version == "v1.0-trainval":
        create_groundtruth_database(args.root_path, info_path=info_path, nsweeps=args.nsweeps)
