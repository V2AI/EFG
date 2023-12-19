import argparse
import io
import json
import os
import pickle
from functools import reduce

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, transform_matrix

from efg.data.datasets.nuscenes.utils import general_to_detection, read_file, read_sweep
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


def _sensor_to_ref_channel(nusc: NuScenes, sensor_data_token: str, ref_data_token: str) -> dict:
    """
    Transform from a particular sensor coordinate frame to a reference channel.

    Args:
        :param nusc: A NuScenes instance.
        :param sensor_data_token: The token of the input sensor data.
        :param ref_data_token: The token of the reference sensor data.

    Returns:
        A dictionary with the following fields:
            - "ref_from_sensor": A 4x4 transformation matrix from sensor to reference frame.
            - "sensor_from_car": A 4x4 transformation matrix from ego car frame to sensor frame.
            - "car_from_global": A 4x4 transformation matrix from global frame to ego car frame.
    """

    # sensor -> ego -> global -> ego' -> ref

    # reference sensor data
    ref_sd_record = nusc.get("sample_data", ref_data_token)
    ref_data_path = nusc.get_sample_data_path(ref_data_token)
    ref_time = 1e-6 * ref_sd_record["timestamp"]

    # Homogeneous transform from ego car frame to reference frame
    # calibrated_sensor: sensor w.r.t. ego car
    ref_cs_record = nusc.get("calibrated_sensor", ref_sd_record["calibrated_sensor_token"])
    ref_from_car = transform_matrix(ref_cs_record["translation"], Quaternion(ref_cs_record["rotation"]), inverse=True)
    # Homogeneous transformation matrix from global to _current_ ego car frame
    # ego_pose: ego car w.r.t. global
    ref_pose_record = nusc.get("ego_pose", ref_sd_record["ego_pose_token"])
    car_from_global = transform_matrix(
        ref_pose_record["translation"], Quaternion(ref_pose_record["rotation"]), inverse=True
    )

    if sensor_data_token != ref_data_token:
        # current sensor data
        sd_record = nusc.get("sample_data", sensor_data_token)

        # Homogeneous transformation matrix from global
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
        # ego pose
        global_from_car = transform_matrix(
            pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False
        )

        # Homogeneous transformation matrix from ego car frame to sensor frame
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        # camera extrinsic
        car_from_current = transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False)

        ref_from_current = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        sensor_data_path = nusc.get_sample_data_path(sd_record["token"])
        time_lag = ref_time - 1e-6 * sd_record["timestamp"]

        sensor_info = {
            "sd_token": sensor_data_token,
            "data_path": sensor_data_path,
            "modality": sd_record["sensor_modality"],
            "transform_matrix": ref_from_current,
            "time_lag": time_lag,
            "timestamp": 1e-6 * sd_record["timestamp"],
            "global_from_car": global_from_car,
            "car_from_current": car_from_current,
        }

        if sd_record["sensor_modality"] == "camera":
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            sensor_info["cam_intrinsic"] = cam_intrinsic
            sensor_info["im_width"] = sd_record["width"]
            sensor_info["im_height"] = sd_record["height"]

        return sensor_info
    else:
        return {
            "sd_token": ref_data_token,
            "data_path": ref_data_path,
            "modality": ref_sd_record["sensor_modality"],
            "transform_matrix": np.eye(4),
            "time_lag": 0,
            "timestamp": ref_time,
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
        }


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    sample_timestamp = sample["timestamp"]
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, "pose")
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop("utime")  # useless
    pos = last_pose.pop("pos")
    rotation = last_pose.pop("orientation")
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus)


def _fill_trainval_infos(
    nusc, nusc_canbus, train_scenes, val_scenes, test=False, nsweeps=10, ref_chan="LIDAR_TOP", occ=False, seg=False
):
    """
    Generate the info file for training and validation. Here we store data from all modalities in LIDAR_TOP coordinate.

    Args:
        nusc: A NuScenes instance.
        train_scenes: Scenes for training.
        val_scenes: Scenes for validation.
        test: Whether to generate infos for test set.
        nsweeps: Number of sweeps for lidar.
        ref_chan: The reference channel of the dataset, e.g. 'LIDAR_TOP'.

    Returns:
        train_nusc_infos: Info for training set.
        val_nusc_infos: Info for validation set.
    """

    train_nusc_infos = []
    val_nusc_infos = []

    root_path = nusc.dataroot

    if occ:
        occ_annotations = json.load(open(os.path.join(root_path, "occupancy", "annotations.json"), "r"))["scene_infos"]
        occ_annotations["occ_path"] = os.path.join(root_path, "occupancy")
    else:
        occ_annotations = None

    for sample in tqdm(nusc.sample):

        info = {
            "prev": sample["prev"],
            "next": sample["next"],
            "timestamp": 1e-6 * sample["timestamp"],
            "sample_token": sample["token"],
            "scene_token": nusc.get("sample", sample["token"])["scene_token"],
            "ref_chan": ref_chan,
        }
        info["map_location"] = nusc.get("log", nusc.get("scene", info["scene_token"])["log_token"])["location"]

        can_bus = _get_can_bus_info(nusc, nusc_canbus, sample)
        info["CAN_BUS"] = can_bus

        for channel, token in sample["data"].items():
            sd_record = nusc.get("sample_data", token)
            sensor_modality = sd_record["sensor_modality"]
            if sensor_modality == "camera":
                info[channel] = {}
                info[channel].update(_sensor_to_ref_channel(nusc, token, sample["data"][ref_chan]))
            elif sensor_modality == "lidar":
                info[channel] = {}
                info[channel].update(_sensor_to_ref_channel(nusc, token, sample["data"][ref_chan]))
                # Aggregate current and previous sweeps.
                current_sd_rec = nusc.get("sample_data", token)
                sweeps = []
                while len(sweeps) < nsweeps - 1:
                    if current_sd_rec["prev"] == "":
                        if len(sweeps) == 0:
                            sweeps.append(_sensor_to_ref_channel(nusc, token, sample["data"][ref_chan]))
                        else:
                            sweeps.append(sweeps[-1])
                    else:
                        token = current_sd_rec["prev"]
                        current_sd_rec = nusc.get("sample_data", token)
                        sweeps.append(_sensor_to_ref_channel(nusc, token, sample["data"][ref_chan]))
                info[channel]["sweeps"] = sweeps
                assert len(info[channel]["sweeps"]) == nsweeps - 1, \
                    f"sweep {current_sd_rec['token']} has {len(info[channel]['sweeps'])} sweeps, expect #{nsweeps-1}."

            else:
                info[channel] = {}
                info[channel].update(_sensor_to_ref_channel(nusc, token, sample["data"][ref_chan]))

        if not test:

            _, ref_boxes, _ = get_sample_data(nusc, sample["data"][ref_chan])
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
                [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations], dtype=bool
            ).reshape(-1)

            info["annotations"] = {}
            info["annotations"]["gt_boxes"] = gt_boxes[mask].astype(np.float32)
            info["annotations"]["gt_box_tokens"] = tokens[mask]
            info["annotations"]["gt_names"] = np.array([general_to_detection[name] for name in names])[mask]
            info["annotations"]["gt_names_raw"] = np.array(names)[mask]

            assert len(annotations) == len(gt_boxes) == len(velocity)

        # occupancy
        if occ:
            scene_name = nusc.get("scene", sample["scene_token"])["name"]
            sample_occ = occ_annotations[scene_name][sample["token"]]
            info["annotations"]["occ_path"] = os.path.join(occ_annotations["occ_path"], sample_occ["gt_path"])

        if seg:
            # lidar-seg
            lidar_seg = nusc.get("lidarseg", sample["data"]["LIDAR_TOP"])
            panoptic = nusc.get("panoptic", sample["data"]["LIDAR_TOP"])
            info["annotations"]["lidarseg"] = lidar_seg
            info["annotations"]["panoptic"] = panoptic

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def create_nuscenes_infos(root_path, version="v1.0-trainval", nsweeps=10, occ=False, seg=False, client=None, nproc=8):

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

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_canbus = NuScenesCanBus(dataroot=root_path)

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

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_canbus, train_scenes, val_scenes, test, nsweeps=nsweeps, ref_chan="LIDAR_TOP", occ=occ, seg=seg,
    )

    if "s3" not in root_path:
        if test:
            print(f"test sample: {len(train_nusc_infos)}")
            with open(os.path.join(root_path, f"infos_test_{nsweeps:02d}sweeps_with_cam_reorg.pkl"), "wb") as f:
                pickle.dump(train_nusc_infos, f)
        else:
            print(f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}")
            with open(os.path.join(root_path, f"infos_train_{nsweeps:02d}sweeps_with_cam_reorg.pkl"), "wb") as f:
                pickle.dump(train_nusc_infos, f)
            with open(os.path.join(root_path, f"infos_val_{nsweeps:02d}sweeps_with_cam_reorg.pkl"), "wb") as f:
                pickle.dump(val_nusc_infos, f)
    else:
        if test:
            print(f"test sample: {len(train_nusc_infos)}")
            client.put(
                root_path.replace("nuScenes", f"nuScenesEFG{args.nsweeps:02d}F")
                + f"/infos_test_{nsweeps:02d}sweeps_with_cam_reorg.pkl",
                pickle.dumps(train_nusc_infos),
            )
        else:
            print(f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}")
            client.put(
                root_path.replace("nuScenes", f"nuScenesEFG{args.nsweeps:02d}F")
                + f"/infos_train_{nsweeps:02d}sweeps_with_cam_reorg.pkl",
                pickle.dumps(train_nusc_infos),
            )
            client.put(
                root_path.replace("nuScenes", f"nuScenesEFG{args.nsweeps:02d}F")
                + f"/infos_val_{nsweeps:02d}sweeps_with_cam_reorg.pkl",
                pickle.dumps(val_nusc_infos),
            )


def create_groundtruth_database(
    data_path,
    info_path=None,
    used_classes=(
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ),
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    nsweeps=1,
    client=None,
):
    root_path = data_path

    if "s3" not in root_path:
        if db_path is None:
            db_path = os.path.join(root_path, f"gt_database_train_{nsweeps:02d}sweeps_with_cam_reorg")
        if dbinfo_path is None:
            dbinfo_path = os.path.join(root_path, f"gt_database_train_{nsweeps:02d}sweeps_with_cam_reorg_infos.pkl")
        if not os.path.exists(db_path):
            os.makedirs(db_path)
    else:
        if db_path is None:
            db_path = (
                root_path.replace("nuScenes", f"nuScenesEFG{nsweeps:02d}F")
                + f"/gt_database_train_{nsweeps:02d}sweeps_with_cam_reorg"
            )
        if dbinfo_path is None:
            dbinfo_path = (
                root_path.replace("nuScenes", f"nuScenesEFG{nsweeps:02d}F")
                + f"/gt_database_train_{nsweeps:02d}sweeps_with_cam_reorg_infos.pkl"
            )

    # nuscenes dataset setting
    point_features = 5

    all_db_infos = {}
    group_counter = 0

    if "s3" not in root_path:
        with open(info_path, "rb") as f:
            dataset_infos_all = pickle.load(f)
    else:
        info_pkl_bytes = client.get(info_path)
        dataset_infos_all = pickle.load(io.BytesIO(info_pkl_bytes))

    ref_chan = "LIDAR_TOP"
    for info in tqdm(dataset_infos_all):
        info_lidar = info[ref_chan]
        lidar_path = info_lidar["data_path"]
        if "s3" in root_path:
            lidar_path = lidar_path.replace("datasets/nuscenes", "s3://Datasets/nuScenes")

        points = read_file(lidar_path)

        # points[:, 3] /= 255
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (nsweeps - 1) <= len(info_lidar["sweeps"]), "nsweeps {} should not greater than list length {}.".format(
            nsweeps, len(info_lidar["sweeps"])
        )

        for i in range(nsweeps - 1):
            points_sweep, times_sweep = read_sweep(info_lidar["sweeps"][i])
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
            "boxes": info["annotations"]["gt_boxes"],
            "names": info["annotations"]["gt_names"],
            "tokens": info["annotations"]["gt_box_tokens"],
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

                filename = f"{info_lidar['sd_token']}_{names[i]}_{i}.bin"
                if "s3" not in db_path:
                    dirpath = os.path.join(db_path, names[i])

                    if not os.path.exists(dirpath):
                        os.makedirs(dirpath)

                    filepath = os.path.join(db_path, names[i], filename)

                    with open(filepath, "w") as f:
                        try:
                            gt_points[:, :point_features].tofile(f)
                        except:
                            print("process {} files".format(info_lidar["sd_token"]))
                            break
                else:
                    filepath = db_path + "/" + names[i] + "/" + filename
                    client.put(filepath, pickle.dumps(gt_points[:, :point_features]))

                if relative_path:
                    db_dump_path = os.path.join(
                        f"gt_database_train_{nsweeps:02d}sweeps_with_cam_reorg", names[i], filename
                    )
                else:
                    db_dump_path = filepath

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "sd_token": info_lidar["sd_token"],
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

    if "s3" not in dbinfo_path:
        with open(dbinfo_path, "wb") as f:
            pickle.dump(all_db_infos, f)
    else:
        client.put(dbinfo_path, pickle.dumps(all_db_infos))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("nuScenes dataset preparation")
    parser.add_argument("--root-path", default=None)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--nsweeps", default=1, type=int)
    parser.add_argument("--occ", action="store_true")
    parser.add_argument("--seg", action="store_true")

    args = parser.parse_args()

    if "s3" in args.root_path:
        from petrel_client.client import Client

        client = Client("~/.petreloss.conf")
    else:
        client = None

    create_nuscenes_infos(args.root_path, args.version, args.nsweeps, args.occ, args.seg, client)

    if "s3" not in args.root_path:
        info_path = os.path.join(args.root_path, f"infos_train_{args.nsweeps:02d}sweeps_with_cam_reorg.pkl")
    else:
        info_path = (
            args.root_path.replace("nuScenes", f"nuScenesEFG{args.nsweeps:02d}F")
            + f"/infos_train_{args.nsweeps:02d}sweeps_with_cam_reorg.pkl"
        )

    if args.version == "v1.0-trainval":
        create_groundtruth_database(args.root_path, info_path=info_path, nsweeps=args.nsweeps, client=client)
