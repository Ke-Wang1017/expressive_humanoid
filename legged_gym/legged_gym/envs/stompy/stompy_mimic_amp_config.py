from legged_gym.envs.stompy.stompy_mimic_config import StompyMimicCfg, StompyMimicCfgPPO

class StompyMimicAMPCfg( StompyMimicCfg ):
    class amp():
        num_obs_steps = 2
        num_obs_per_step = 20 + 3 + 3 + 3 + 12*3

class StompyMimicAMPCfgPPO( StompyMimicCfgPPO ):
    class runner( StompyMimicCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimicAMP"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
    
    class amp():
        amp_input_dim = StompyMimicAMPCfg.amp.num_obs_steps * StompyMimicAMPCfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [1024, 512]

        amp_replay_buffer_size = 1000000
        amp_demo_buffer_size = 200000
        amp_demo_fetch_batch_size = 512
        amp_learn_batch_size = 4096
        amp_learning_rate = 1.e-4

        amp_reward_coef = 4.0

        amp_grad_pen = 5