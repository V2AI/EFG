import numpy as np
import copy
import numpy as np


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)


WAYMO_TRACKING_NAMES = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]


class PubTracker(object):
    def __init__(self, max_dist={}):
        self.WAYMO_CLS_VELOCITY_ERROR = max_dist
        self.WAYMO_TRACKING_NAMES = WAYMO_TRACKING_NAMES

    def reset(self, id, tracks):
        self.id_count = id
        self.tracks = tracks

    def step_centertrack(self, results, time_lag, frame_id):
        if len(results) == 0:
            self.tracks = []
            return []
        else:
            temp = []
            for det in results:
                # filter out classes not evaluated for tracking
                if det["detection_name"] not in self.WAYMO_TRACKING_NAMES:
                    print("filter {}".format(det["detection_name"]))
                    continue

                det["ct"] = np.array(det["translation"][:2])
                det["tracking"] = np.array(det["velocity"][:2]) * -1 * time_lag
                det["label_preds"] = (
                    self.WAYMO_TRACKING_NAMES.index(det["detection_name"]) + 1
                )
                temp.append(det)

            results = temp

        N = len(results)
        M = len(self.tracks)
        # N X 2
        if "tracking" in results[0]:
            dets = np.array(
                [det["ct"] + det["tracking"].astype(np.float32) for det in results],
                np.float32,
            )
        else:
            dets = np.array([det["ct"] for det in results], np.float32)

        item_cat = np.array([item["label_preds"] for item in results], np.int32)  # N
        track_cat = np.array(
            [track["label_preds"] for track in self.tracks], np.int32
        )  # M

        max_diff = np.array(
            [self.WAYMO_CLS_VELOCITY_ERROR[box["detection_name"]] for box in results],
            np.float32,
        )

        tracks = np.array(
            [pre_det["ct"] for pre_det in self.tracks], np.float32
        )  # M x 2

        if len(tracks) > 0:  # NOT FIRST FRAME
            dist = ((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(
                axis=2
            )  # N x M
            dist = np.sqrt(dist)  # absolute distance in meter

            invalid = (
                (dist > max_diff.reshape(N, 1))
                + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))
            ) > 0

            dist = dist + invalid * 1e18
            matched_indices = greedy_assignment(copy.deepcopy(dist))
        else:  # first few frame
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)

        unmatched_dets = [
            d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])
        ]

        unmatched_tracks = [
            d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])
        ]

        matches = matched_indices

        ret = []
        for m in matches:
            track = results[m[0]]
            track["tracking_id"] = self.tracks[m[1]]["tracking_id"]
            track["age"] = 1
            track["active"] = self.tracks[m[1]]["active"] + 1
            ret.append(track)
        return ret
