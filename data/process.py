""" # noqa: E501
Re-create the synced and aligned experimental data of the experiments done in Berlin in June 23.
Go into the data folder and execute this file.
"""
import json
import os

import joblib
import numpy as np
import qmt
import tree
from x_xy.subpkgs import exp
from x_xy.subpkgs import omc

output_file_ending = ""

_marker_closest_to_rigid_imu = {"seg1": 4, "seg5": 4, "seg2": 3, "seg3": 3, "seg4": 2}

# if experiment is not present, then alignment will be skipped
# this motion phase should not contain too crazy motion; it will just
# make the acceleration estimate "hide" gravity
_alignment_timings_motion = {
    "S_04": "slow",
    "S_06": "slow1",
    "S_07": "slow_fast_mix",
    "S_08": "slow1",
    "S_09": "slow_global",
    "S_10": "pickandplace",
    "S_12": "slow1",
    "S_13": "slow_fast_mix",
    "S_14": "slow",
    "S_15": "slow_global",
    "S_16": "gait_slow",
    "T_01": ("slow", "shaking"),
}


def _alignment_timings(exp_id: str) -> tuple[float, float]:
    timings = exp.load_timings(exp_id)
    timings_list = list(timings.keys())
    motion = _alignment_timings_motion[exp_id]
    if isinstance(motion, str):
        next_motion = timings_list[timings_list.index(motion) + 1]
    else:
        motion, next_motion = motion
    return timings[motion], timings[next_motion]


def _segment_names_in_experiment(exp_id: str) -> list[str]:
    all = set(["seg1", "seg2", "seg3", "seg4", "seg5"])

    if exp_id == "S_04":
        all -= set(["seg1", "seg5"])
    elif exp_id == "D_01":
        all -= set(["seg3"])
    else:
        pass

    return list(all)


def _imu_names_setup_file(exp_id: str) -> list[str]:
    if exp_id == "D_01":
        return ["imu_rigid"]
    return ["imu_rigid", "imu_flex"]


def _get_alignment(data: dict, exp_id: str, hz_omc, hz_imu):
    hz_alignment = 120.0
    hz_in = omc.hz_helper(
        list(data.keys()),
        hz_imu=hz_imu,
        hz_omc=hz_omc,
        imus=_imu_names_setup_file(exp_id),
    )
    data = omc.resample(data, hz_in, hz_alignment, vecinterp_method="cubic")
    data = omc.crop_tail(data, hz_alignment)
    t1, t2 = _alignment_timings(exp_id)
    data = exp.exp._crop_sequence(data, 1 / hz_alignment, t1, t2)

    acc, gyr, mag, q, pos, names = [], [], [], [], [], []
    for seg_name, seg_data in data.items():
        imu = seg_data["imu_rigid"]
        acc.append(imu["acc"])
        gyr.append(imu["gyr"])
        mag.append(imu["mag"])
        q.append(seg_data["quat"])
        pos.append(seg_data[f"marker{_marker_closest_to_rigid_imu[seg_name]}"])
        names.append(seg_name)

    info = qmt.alignOptImu(
        gyr, acc, mag, q, pos, rate=hz_alignment, names=names, params=dict(fast=True)
    )

    def _np_tolist(leaf):
        if isinstance(leaf, np.ndarray):
            return leaf.tolist()
        return leaf

    with open(
        f"alignment_infos/alignment_info_{exp_id}{output_file_ending}.json", "w"
    ) as file:
        json.dump(tree.map_structure(_np_tolist, info), file, indent=1)
    joblib.dump(
        info, f"alignment_infos/alignment_info_{exp_id}{output_file_ending}.joblib"
    )

    qEOpt2EImu_euler_deg = info["qEOpt2EImu_euler_deg"]
    qImu2Seg_euler_deg = {seg_name: dict() for seg_name in data}
    for seg_name in data:
        qImu2Seg_euler_deg[seg_name]["imu_rigid"] = info[
            f"qImu2Seg_{seg_name}_euler_deg"
        ]
    print(qImu2Seg_euler_deg)
    return qEOpt2EImu_euler_deg, qImu2Seg_euler_deg


def to_joblib(exp_id: str):
    hz_omc = exp.load_hz_omc(exp_id)

    output_path = f"{exp_id}{output_file_ending}.joblib"
    _exp_folder = (
        "/Users/simon/Documents/data/berlin_02_06_23/berlin_02_06_23/experiments"
    )
    path_optitrack_file = f"{_exp_folder}/{exp_id}/optitrack/{exp_id}_{hz_omc}Hz.csv"
    path_imu_folder = f"{_exp_folder}/{exp_id}/imu"
    setup_file = (
        "/Users/simon/Documents/PYTHON/x_xy_v2/x_xy/subpkgs/exp/setups/setup.json"
    )

    hz_omc = float(hz_omc)
    hz_imu = float(omc.utils.autodetermine_imu_freq(path_imu_folder))

    # perform sync
    data, imu_sync_offset = omc.read_omc(
        path_marker_imu_setup_file=setup_file,
        path_optitrack_file=path_optitrack_file,
        path_imu_folder=path_imu_folder,
        imu_names_setup_file=_imu_names_setup_file(exp_id),
        segment_names_setup_file=_segment_names_in_experiment(exp_id),
    )

    do_alignment = exp_id in _alignment_timings_motion
    if do_alignment:
        qEOpt2EImu_euler_deg, qImu2Seg_euler_deg = _get_alignment(
            data, exp_id, hz_omc, hz_imu
        )

        # aligned
        data, _ = omc.read_omc(
            path_marker_imu_setup_file=setup_file,
            path_optitrack_file=path_optitrack_file,
            path_imu_folder=path_imu_folder,
            imu_sync_offset=imu_sync_offset,
            qEOpt2EImu_euler_deg=qEOpt2EImu_euler_deg,
            qImu2Seg_euler_deg=qImu2Seg_euler_deg,
            imu_names_setup_file=_imu_names_setup_file(exp_id),
            segment_names_setup_file=_segment_names_in_experiment(exp_id),
        )

    hz_in = omc.hz_helper(
        list(data.keys()),
        hz_imu=hz_imu,
        hz_omc=hz_omc,
        imus=_imu_names_setup_file(exp_id),
    )
    # croptail
    data = omc.crop_tail(data, hz_in, strict=False)

    joblib.dump(data, output_path)

    return data


def main():
    _already_processed = os.listdir(".")
    exp_ids = [
        "S_04",
        "S_06",
        "S_07",
        "S_08",
        "S_09",
        "S_10",
        "S_12",
        "S_13",
        "S_14",
        "S_15",
        "S_16",
        # "D_01",
        "T_01",
    ]
    for exp_id in exp_ids:
        if (exp_id + ".joblib") in _already_processed:
            continue

        to_joblib(exp_id)


if __name__ == "__main__":
    main()
