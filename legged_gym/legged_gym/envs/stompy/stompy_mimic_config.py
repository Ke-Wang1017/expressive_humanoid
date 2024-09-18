# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class StompyMimicCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096

        n_demo_steps = 2
        n_demo = 9 + 3 + 3 + 3 +6*3  #observe height
        interval_demo_steps = 0.1

        n_scan = 0 #132 terrain scan
        n_priv = 3
        n_priv_latent = 4 + 1 + 18*2
        n_proprio = 3 + 2 + 2 + 18*3 + 2 # one hot
        history_len = 10

        prop_hist_len = 4
        n_feature = prop_hist_len * n_proprio

        num_observations = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv

        episode_length_s = 50 # episode length in seconds
        num_actions = 18
        
        num_policy_actions = 18
    
    class motion:
        motion_curriculum = True
        motion_type = "yaml"
        motion_name = "motions_autogen_walk.yaml"

        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        no_keybody = False
        regen_pkl = False

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10


    class terrain( LeggedRobotCfg.terrain ):
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        height = [0., 0.04]
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.63] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "L_hip_y": -0.157,
            "L_hip_x": 0.0394,
            "L_hip_z": 0.0628,
            "L_knee": 0.441,
            "L_ankle_y": -0.258,
            "L_shoulder_y": 0.0,
            "L_shoulder_x": 0.0,
            "L_shoulder_z": 0.0,
            "L_elbow_x": 0.0,
            "R_hip_y": -0.22,
            "R_hip_x": 0.026,
            "R_hip_z": 0.0314,
            "R_knee": 0.441,
            "R_ankle_y": -0.223,
            "R_shoulder_y": 0.0,
            "R_shoulder_x": 0.0,
            "R_shoulder_z": 0.0,
            "R_elbow_x": 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'L_hip_y': 120,
                     'L_hip_x': 60,
                     'L_hip_z': 60,
                     'L_knee': 120,
                     'L_ankle_y': 17,
                     'R_hip_y': 120,
                     'R_hip_x': 60,
                     'R_hip_z': 60,
                     'R_knee': 120,
                     'R_ankle_y': 17,
                     'shoulder': 40,
                     "elbow":40,
                     }  # [N*m/rad]
        damping = {  'L_hip_y': 6,
                     'L_hip_x': 3,
                     'L_hip_z': 3,
                     'L_knee': 6,
                     'L_ankle_y': 1,
                     'R_hip_y': 6,
                     'R_hip_x': 3,
                     'R_hip_z': 3,
                     'R_knee': 6,
                     'R_ankle_y': 1,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class normalization( LeggedRobotCfg.normalization):
        clip_actions = 10

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/stompypro/robot.urdf'
        torso_name = "trunk"
        foot_name = "foot"
        foot_name_list = ["L_foot",
                          "R_foot"]
        hip_joint_list = ["L_hip_y", "R_hip_y"]
        penalize_contacts_on = ["shoulder", "arm", "clav", "scapula"]
        terminate_after_contacts_on = ["trunk", "thigh", "calf" ]#]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            alive = 1
            # tracking_demo_goal_vel = 1.0
            # tracking_mul = 6
            tracking_lin_vel = 6
            # stand_still = 3
            # tracking_goal_vel = 4


            tracking_demo_yaw = 1
            tracking_demo_roll_pitch = 1
            orientation = -2
            tracking_demo_dof_pos = 3
            # tracking_demo_dof_vel = 1.0
            tracking_demo_key_body = 2
            # tracking_demo_height = 1  # useful if want better height tracking
            
            # tracking_demo_lin_vel = 1
            # tracking_demo_ang_vel = 0.5
            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.4
            # orientation = -1.
            dof_acc = -3e-7
            collision = -10.
            action_rate = -0.1
            # delta_torques = -1.0e-7
            # torques = -1e-5
            energy = -1e-3
            # hip_pos = -0.5
            dof_error = -0.1
            feet_stumble = -2
            # feet_edge = -1
            feet_drag = -0.1
            dof_pos_limits = -10.0
            feet_air_time = 10
            feet_height = 2
            feet_force = -3e-3

        only_positive_rewards = False
        clip_rewards = True
        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]
    
    class noise():
        add_noise = True
        noise_scale = 0.5 # scales other values
        class noise_scales():
            dof_pos = 0.01
            dof_vel = 0.15
            ang_vel = 0.3
            imu = 0.2
    

class StompyMimicCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimic"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPOMimic'
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False
        text_feat_input_dim = StompyMimicCfg.env.n_feature
        text_feat_output_dim = 16
        feat_hist_len = StompyMimicCfg.env.prop_hist_len
        # actor_hidden_dims = [1024, 512]
        # critic_hidden_dims = [1024, 512]
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005

    class estimator:
        train_with_estimated_states = False
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = StompyMimicCfg.env.n_priv
        priv_start = StompyMimicCfg.env.n_feature + StompyMimicCfg.env.n_proprio + StompyMimicCfg.env.n_demo + StompyMimicCfg.env.n_scan
        
        prop_start = StompyMimicCfg.env.n_feature
        prop_dim = StompyMimicCfg.env.n_proprio

class StompyMimicDistillCfgPPO( StompyMimicCfgPPO ):
    class distill:
        num_demo = 3
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]

        num_mini_batches = 4