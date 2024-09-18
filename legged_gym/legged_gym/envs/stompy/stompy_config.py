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


class StompyCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.875] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
                'right shoulder pitch': 0.0987, 
                'right shoulder yaw': 0.0264, 
                'right shoulder roll': -2.87,
                'right elbow pitch': 2.85,
                'right wrist roll':  -0.314,
                'left hip pitch':  -0.04485,
                'left hip yaw':  1.57102,
                'left hip roll':  1.53816,
                'left knee pitch':  0.14658,
                'left ankle pitch':  -1.49738,
                'right hip pitch':  3.14,
                'right hip yaw':  3.15,
                'right hip roll':  -3.18,
                'right knee pitch':  0,
                'right ankle pitch':  -0.19,
                'left shoulder pitch':  2.06,
                'left shoulder yaw':  -0.656,
                'left shoulder roll':  1.34,
                'left elbow pitch':  -0.241,
                'left wrist roll': 0.0
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 80.}  # [N*m/rad]
        damping = {'joint': 1}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/stompypro/robot.urdf'
        torso_name = "link_upper_half_assembly_1_torso_bottom_half_1"
        foot_name = "foot_pad_1"
        penalize_contacts_on = ["shoulder", "elbow", "hip"]
        terminate_after_contacts_on = ["torso", "knee", "leg" ]#]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
    
    class amp():
        num_obs_steps = 10
        num_obs_per_step = 20 + 3 # 19 joint angles + 3 base ang vel

class StompyCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005

    class amp():
        amp_input_dim = StompyCfg.amp.num_obs_steps * StompyCfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [512, 256]

        amp_replay_buffer_size = 10000
        amp_demo_buffer_size = 10000
        amp_demo_fetch_batch_size = 512
        amp_learning_rate = 1.e-4

        amp_reward_coef = 2.0