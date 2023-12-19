import numpy as np
from pyquaternion import Quaternion

from efg.utils.file_io import PathManager

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[:, 0]) < radius
    y_filt = np.abs(points[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return not_close


def read_file(path, num_point_feature=4):
    try:
        bytes_data = PathManager.open(path, "rb").read()
        points = np.copy(np.frombuffer(bytes_data, np.float32))
        s = points.shape[0]
        if s % 5 != 0:
            points = points[: s - (s % 5)]
        points = points.reshape(-1, 5)[:, :num_point_feature]
    except Exception:
        print(f"Read Failed: {path}")
        points = None

    return points


def read_sweep(sweep):
    min_distance = 1.0

    sweep_path = sweep["data_path"]
    points_sweep = read_file(sweep_path)
    if points_sweep is None:
        return None, None

    not_close = remove_close(points_sweep, min_distance)  # remove in its local coordinate
    points_sweep = points_sweep[not_close].T

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(np.vstack((points_sweep[:3, :], np.ones(nbr_points))))[
            :3, :
        ]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


def extract_translation_rotation(T: np.ndarray, inverse: bool = False, combined: bool = False):
    """
    Extract translation and rotation (as a quaternion) from a 4x4 transformation matrix.
    :param T: <np.float32: 4, 4>. Transformation matrix.
    :param inverse: Whether the transformation matrix is the inverse.
    :return: translation (3x1 ndarray), rotation (Quaternion).
    """
    if inverse:
        # If the transformation matrix is an inverse, compute the original rotation and translation
        rotation_matrix = T[:3, :3].T
        translation = -rotation_matrix.dot(T[:3, 3])
    else:
        rotation_matrix = T[:3, :3]
        translation = T[:3, 3]

    if not combined:
        # Convert rotation matrix to quaternion
        rotation_quaternion = Quaternion(matrix=rotation_matrix)
        return {
            "translation": translation,
            "rotation": rotation_quaternion
        }
    else:
        T_prime = np.eye(4)
        T_prime[:3, :3] = rotation_matrix
        T_prime[:3, 3] = translation

        return T_prime


def euler_to_quaternion(yaw, pitch, roll):
    """
    Convert yaw, pitch, roll (Euler angles) to a Quaternion.
    :param yaw: Rotation about the Z-axis.
    :param pitch: Rotation about the Y-axis.
    :param roll: Rotation about the X-axis.
    :return: Quaternion representation.
    """
    return Quaternion(axis=[0, 0, 1], angle=yaw) * Quaternion(axis=[0, 1, 0], angle=pitch) * \
        Quaternion(axis=[1, 0, 0], angle=roll)
