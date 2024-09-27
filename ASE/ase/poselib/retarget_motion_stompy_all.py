from isaacgym.torch_utils import *
import torch
import json
import numpy as np
import sys
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
import yaml
import os
import multiprocessing
from tqdm import tqdm

"""
This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
  - source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
  - target_motion_path: path to save the retargeted motion to
  - source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
  - target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
  - joint_mapping: mapping of joint names from source to target
  - rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
  - scale: scale offset from source to target skeleton
"""


def process(i, motion_name, source_tpose, target_tpose, source_motion, target_motion, retarget_data):
    source_motion_path = os.path.join(source_motion, motion_name + ".npy")
    target_motion_path = os.path.join(target_motion, motion_name + ".npy")
    try:
        source_motion = SkeletonMotion.from_file(source_motion_path)
    except:
        print("failed to load motion: ", source_motion_path)
        return
    # parse data from retarget config
    joint_mapping = retarget_data["joint_mapping"]
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

    print(f"run retargeting {i}: {motion_name}")
    # run retargeting
    target_motion = source_motion.retarget_to_by_tpose(
      joint_mapping=retarget_data["joint_mapping"],
      source_tpose=source_tpose,
      target_tpose=target_tpose,
      rotation_to_target_skeleton=rotation_to_target_skeleton,
      scale_to_target_skeleton=retarget_data["scale"]
    )
    frame_beg = retarget_data["trim_frame_beg"]
    frame_end = retarget_data["trim_frame_end"]
    if (frame_beg == -1):
        frame_beg = 0
        
    if (frame_end == -1):
        frame_end = target_motion.local_rotation.shape[0]
        
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
      
    tar_global_pos = target_motion.global_translation[frame_beg:frame_end, ...]
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h

    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # save retargeted motion
    target_motion.to_file(target_motion_path)

    return

def save_all():
    retarget_data_path = "data/configs/retarget_cmu_to_stompy.json"
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)

    source_tpose = SkeletonState.from_file(retarget_data["source_tpose"])
    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])
    source_motion = retarget_data["source_motion"]
    target_motion = retarget_data["target_motion_path"]

    with open("data/configs/motions_debug.yaml", 'r') as f:
        motions_list = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
    all_motion_names =[]
    for motion_entry in motions_list.keys():
        if motion_entry == "root":
            continue
        target_motion_file = os.path.join(target_motion, motion_entry + ".npy")
        # if os.path.exists(target_motion_file):
        #     print("Already exists, skip: ", motion_entry)
        #     continue
        all_motion_names.append(motion_entry)
    all_motion_names.sort()
    print(len(all_motion_names))
    # Number of processes
    n_workers = multiprocessing.cpu_count()

    # Create a pool of worker processes
    with multiprocessing.Pool(n_workers) as pool:
        # Using starmap to pass multiple arguments to the process_file function
        list(tqdm(pool.starmap(process, [(i, motion_name, source_tpose, target_tpose, source_motion, target_motion, retarget_data) for i, motion_name in enumerate(all_motion_names)]), total=len(all_motion_names)))
    
if __name__ == '__main__':
    # main()
    save_all()