import numpy as np

from roboseq.utils.torch_jit_utils import *
from roboseq.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

import torch
import logging
from PIL import Image
import open3d as o3d

def iprint(*strings):
    print(strings)
    exit()
    return


class ArnieDrawer(BaseTask):
    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless,
        agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]],
        is_multi_agent=False,
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.arnie_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)

        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (control_freq_inv * self.sim_params.dt))
            )

        self.object_type = self.cfg["env"]["objectType"]
        self.ignore_z = False

        self.obs_type = self.cfg["env"]["observationType"]
        self.num_obs_dict = {"full_state": 453} #453
        self.num_hand_obs = 72 + 95 + 26 + 6
        self.up_axis = "z"
        self.fingertips = [
            "left_ffdistal",
            "left_mfdistal",
            "left_rfdistal",
            "left_lfdistal",
            "left_thdistal",
            "right_ffdistal",
            "right_mfdistal",
            "right_rfdistal",
            "right_lfdistal",
            "right_thdistal",
        ]
        self.num_fingertips = 10

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        self.num_agents = 1
        self.cfg["env"]["numActions"] = 62 #62

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )  # (2,13)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # (68,2)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )  # (88,13)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)  # (68,)

        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_arnie_dofs + self.num_object_dofs
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.arnie_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_arnie_dofs
        ]
        self.arnie_dof_pos = self.arnie_dof_state[..., 0]
        self.arnie_dof_vel = self.arnie_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :,
            self.num_arnie_dofs : self.num_arnie_dofs
            + self.num_object_dofs,
        ]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        self.arnie_default_dof_pos = to_torch(
            [
                -0.4,
                1.57,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                1.57,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=torch.float,
            device=self.device,
        )  # self.arnie_default_dof_pos = torch.zeros(self.num_arnie_dofs, dtype=torch.float, device=self.device)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.global_indices = torch.arange(
            self.num_envs * 3, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        arnie_start_pose = gymapi.Transform()
        arnie_start_pose.p = gymapi.Vec3(2, 0, 0)
        arnie_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 3.14159)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(1, 0, 0.425)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0,0)
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(1, 0, 0.2)
        table_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0,0)

        asset_root = "../assets"
        arnie_asset_file = "urdf/ArnieHIT/arnie_fxd.urdf"
        object_asset_file = "urdf/cup/cup.urdf"
        table_asset_file = "urdf/square_table.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = True
        asset_options.use_mesh_materials = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000
        asset_options.vhacd_params.concavity = 0.0025
        asset_options.vhacd_params.alpha = 0.04
        asset_options.vhacd_params.beta = 1.0
        asset_options.vhacd_params.convex_hull_downsampling = 4
        asset_options.vhacd_params.max_num_vertices_per_ch = 256

        arnie_asset = self.gym.load_asset(
            self.sim, asset_root, arnie_asset_file, asset_options
        )
        object_asset = self.gym.load_asset(
            self.sim, asset_root, object_asset_file, asset_options
        )
        table_asset = self.gym.load_asset(
            self.sim, asset_root, table_asset_file, asset_options
        )

        arnie_dof_props = self.gym.get_asset_dof_properties(arnie_asset)
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)

        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        self.num_arnie_dofs = self.gym.get_asset_dof_count(arnie_asset)
        self.num_arnie_rbs = self.gym.get_asset_rigid_body_count(arnie_asset)

        self.arnie_dof_lower_limits = []
        self.arnie_dof_upper_limits = []
        self.arnie_dof_default_pos = []
        self.arnie_dof_default_vel = []

        for i in range(self.num_arnie_dofs):
            self.arnie_dof_lower_limits.append(arnie_dof_props["lower"][i])
            self.arnie_dof_upper_limits.append(arnie_dof_props["upper"][i])
            self.arnie_dof_default_pos.append(0.0)
            self.arnie_dof_default_vel.append(0.0)

        self.arnie_dof_lower_limits = to_torch(self.arnie_dof_lower_limits, device=self.device)
        self.arnie_dof_upper_limits = to_torch(self.arnie_dof_upper_limits, device=self.device)
        self.arnie_dof_default_pos = to_torch(self.arnie_dof_default_pos, device=self.device)
        self.arnie_dof_default_vel = to_torch(self.arnie_dof_default_vel, device=self.device)

        self.hand_indices = []
        self.object_indices = []
        self.env_indices = []

        self.object_init_state = []

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(arnie_asset, name)
            for name in self.fingertips
        ]

        self.left_hand_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"left_base_link")
        self.left_thumb_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"left_Thumb_Phadist")
        self.left_index_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"left_Index_Phadist")
        self.right_hand_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"right_base_link")
        self.right_thumb_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"right_Thumb_Phadist")
        self.right_index_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"right_Index_Phadist")

        self.drawer_top_handle = self.gym.find_asset_rigid_body_index(object_asset,"drawer_handle_top")
        self.drawer_bottom_handle = self.gym.find_asset_rigid_body_index(object_asset,"drawer_handle_bottom")
        self.drawer_left_handle = self.gym.find_asset_rigid_body_index(object_asset,"door_right_nob_link")
        self.drawer_right_handle = self.gym.find_asset_rigid_body_index(object_asset,"door_left_nob_link")
        
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            arnie_actor = self.gym.create_actor(
                env_ptr, arnie_asset, arnie_start_pose, "hand", i, 0, 0
            )
            object_actor = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "cup", i, 0, 0
            )
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 0, 0
            )
            self.object_init_state.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_actor, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)

            hand_idx = self.gym.get_actor_index(
                env_ptr, arnie_actor, gymapi.DOMAIN_SIM
            )
            self.hand_indices.append(hand_idx)
            self.env_indices.append(env_ptr)

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        
        self.hand_indices = to_torch(
            self.hand_indices, dtype=torch.long, device=self.device
        )
        
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_hand_reward(
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.drawer_left_knob_pos,
            self.drawer_right_knob_pos,
            self.drawer_bottom_handle_pos,
            self.drawer_top_handle_pos,
            self.left_hand_pos,
            self.right_hand_pos,
            self.right_hand_ff_pos,
            self.right_hand_th_pos,
            self.left_hand_ff_pos,
            self.left_hand_th_pos,
            self.dist_reward_scale,
            self.actions,
            self.action_penalty_scale,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # Left Knob
        self.drawer_left_knob_pos = self.rigid_body_states[:, self.num_arnie_rbs + self.drawer_left_handle, 0:3]
        self.drawer_left_knob_rot = self.rigid_body_states[:, self.num_arnie_rbs + self.drawer_left_handle, 3:7]
        
        # Right Knob
        self.drawer_right_knob_pos = self.rigid_body_states[:, self.num_arnie_rbs + self.drawer_right_handle, 0:3]
        self.drawer_right_knob_rot = self.rigid_body_states[:, self.num_arnie_rbs + self.drawer_right_handle, 3:7]

        # Bottom Handle
        self.drawer_bottom_handle_pos = self.rigid_body_states[:, self.num_arnie_rbs + self.drawer_bottom_handle, 0:3]
        self.drawer_bottom_handle_rot = self.rigid_body_states[:, self.num_arnie_rbs + self.drawer_bottom_handle, 3:7]

        # Top Handle
        self.drawer_top_handle_pos = self.rigid_body_states[:, self.num_arnie_rbs + self.drawer_top_handle, 0:3]
        self.drawer_top_handle_rot = self.rigid_body_states[:, self.num_arnie_rbs + self.drawer_top_handle, 3:7]

        self.left_hand_pos = self.rigid_body_states[:, self.left_hand_handle, 0:3] 
        self.left_hand_rot = self.rigid_body_states[:, self.left_hand_handle, 3:7]

        self.right_hand_pos = self.rigid_body_states[:, self.right_hand_handle, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, self.right_hand_handle, 3:7]

        self.left_hand_ff_pos = self.rigid_body_states[:, self.left_index_handle, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, self.left_index_handle, 3:7]

        self.left_hand_th_pos = self.rigid_body_states[:, self.left_thumb_handle, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, self.left_thumb_handle, 3:7]

        self.right_hand_ff_pos = self.rigid_body_states[:, self.right_index_handle, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, self.right_index_handle, 3:7]

        self.right_hand_th_pos = self.rigid_body_states[:, self.right_thumb_handle, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, self.right_thumb_handle, 3:7]
        
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        self.compute_full_state()

    def compute_full_state(self, asymm_obs=False):
        num_ft_states = 13 * int(self.num_fingertips)  # 130
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 60

        self.obs_buf[:, 0 : self.num_arnie_dofs] = unscale(self.arnie_dof_pos, self.arnie_dof_lower_limits, self.arnie_dof_upper_limits)
        self.obs_buf[:, self.num_arnie_dofs : 2 * self.num_arnie_dofs] = ( self.vel_obs_scale * self.arnie_dof_vel )
        self.obs_buf[:, self.num_arnie_dofs * 2 : 3 * self.num_arnie_dofs] = ( self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_arnie_dofs] )

        fingertip_obs_start = self.num_arnie_dofs * 3  # 192
        
        self.obs_buf[:, fingertip_obs_start : fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)

        hand_pose_start = fingertip_obs_start + num_ft_states

        self.obs_buf[:, hand_pose_start : hand_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_pose_start + 3 : hand_pose_start + 4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 4 : hand_pose_start + 5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 5 : hand_pose_start + 6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        self.obs_buf[:, hand_pose_start + 6 : hand_pose_start + 9] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start + 9 : hand_pose_start + 10] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 10 : hand_pose_start + 11] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 11 : hand_pose_start + 12] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 12

        self.obs_buf[:, action_obs_start : action_obs_start + self.num_actions] = self.actions[:, :self.num_actions]

        obj_obs_start = action_obs_start + self.num_actions

        self.obs_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = (self.vel_obs_scale * self.object_angvel)

        self.obs_buf[:, obj_obs_start + 13 : obj_obs_start + 16] = self.drawer_left_knob_pos
        self.obs_buf[:, obj_obs_start + 16 : obj_obs_start + 19] = self.drawer_right_knob_pos
        self.obs_buf[:, obj_obs_start + 19 : obj_obs_start + 22] = self.drawer_bottom_handle_pos
        self.obs_buf[:, obj_obs_start + 22 : obj_obs_start + 25] = self.drawer_top_handle_pos

    def reset(self, env_ids):
        # generate random floats for offsetting
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arnie_dofs + self.num_object_dofs), device=self.device)
        
        # copy initial state
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        
        # Positions at which objects exist
        object_indices = torch.unique(self.object_indices[env_ids].to(torch.int32))
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_indices = torch.unique(torch.cat([hand_indices, object_indices]).to(torch.int32))
        
        # delta to offset position by
        delta_max = self.arnie_dof_upper_limits - self.arnie_dof_default_pos
        delta_min = self.arnie_dof_lower_limits - self.arnie_dof_default_pos
        rand_delta = ( delta_min + (delta_max - delta_min) * rand_floats[:, self.num_object_dofs : self.num_object_dofs + self.num_arnie_dofs] )
        
        # UPDATED POSITION and velocity WITH RANDOM DELTA
        pos = self.arnie_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.arnie_dof_pos[env_ids, :] = pos
        self.arnie_dof_vel[env_ids, :] = (self.arnie_dof_default_vel+ self.reset_dof_vel_noise * rand_floats[:, 0 : self.num_arnie_dofs])
        
        # reset object pos and vel
        self.object_dof_pos[env_ids, :] = to_torch([0], device=self.device)
        self.object_dof_vel[env_ids, :] = to_torch([0], device=self.device)
        
        # prev and current targets set to random position for arms    
        self.prev_targets[env_ids, : self.num_arnie_dofs] = pos
        self.cur_targets[env_ids, : self.num_arnie_dofs] = pos
        
        # prev and current targets set to zero for drawer
        self.prev_targets[env_ids, self.num_arnie_dofs : self.num_arnie_dofs + self.num_object_dofs] = to_torch([0], device=self.device)
        self.cur_targets[env_ids, self.num_arnie_dofs : self.num_arnie_dofs + self.num_object_dofs] = to_torch([0], device=self.device)
        
        # set state to initial state
        self.hand_positions[hand_indices.to(torch.long), :] = self.saved_root_tensor[hand_indices.to(torch.long), 0:3]
        self.hand_orientations[hand_indices.to(torch.long), :] = self.saved_root_tensor[hand_indices.to(torch.long), 3:7]
        self.hand_linvels[hand_indices.to(torch.long), :] = self.saved_root_tensor[hand_indices.to(torch.long), 7:10]
        self.hand_angvels[hand_indices.to(torch.long), :] = self.saved_root_tensor[hand_indices.to(torch.long), 10:13]
        
        # set dof state
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices),
            len(hand_indices),
        )
        
        # set dof position
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(hand_indices),
            len(hand_indices),
        )
        
        # set actor root state
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )
        
        # reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        ### Uses force control
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset(env_ids)

        self.actions = actions.clone().to(self.device)
        
        self.cur_targets[:, :self.num_arnie_dofs] = scale(self.actions[:, :self.num_arnie_dofs], self.arnie_dof_lower_limits, self.arnie_dof_upper_limits)
        self.cur_targets[:, :self.num_arnie_dofs] = self.act_moving_average * self.cur_targets[:, :self.num_arnie_dofs] + (1.0 - self.act_moving_average) * self.prev_targets[:, :self.num_arnie_dofs]
        self.cur_targets[:, :self.num_arnie_dofs] = tensor_clamp(self.cur_targets[:, :self.num_arnie_dofs], self.arnie_dof_lower_limits, self.arnie_dof_upper_limits)

        big_idxs = [9]
        small_idxs = [13]

        for i in big_idxs:
            self.apply_forces[:, i, :] = actions[:, i * 3 : i * 3 + 3] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, i, :] = self.actions[:, i * 3 : i * 3 + 3] * self.dt * self.orientation_scale * 1000

        for i in small_idxs:
            self.apply_forces[:, i, :] = actions[:, i * 3 : i * 3 + 3] * self.dt * self.transition_scale * 10
            self.apply_torque[:, i, :] = self.actions[:, i * 3 : i * 3 + 3] * self.dt * self.orientation_scale * 0.1

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, :self.num_arnie_dofs] = self.cur_targets[:, :self.num_arnie_dofs]
        
    def post_physics_step(self):
        print(self.progress_buf)
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()

@torch.jit.script
def compute_hand_reward(
    rew_buf,
    reset_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    object_rot,
    drawer_left_knob_pos,
    drawer_right_knob_pos,
    drawer_bottom_handle_pos,
    drawer_top_handle_pos,
    left_hand_pos,
    right_hand_pos,
    right_ff_pos,
    right_th_pos,
    left_ff_pos,
    left_th_pos,
    dist_reward_scale: float,
    actions,
    action_penalty_scale: float,
):

    l_t_dist = torch.norm(drawer_top_handle_pos - left_ff_pos, p=2, dim=-1) + torch.norm(drawer_top_handle_pos - left_th_pos, p=2, dim=-1)

    r_t_dist = torch.norm(drawer_top_handle_pos - right_ff_pos, p=2, dim=-1) + torch.norm(drawer_top_handle_pos - right_th_pos, p=2, dim=-1)

    finger_cabinet_dist = (l_t_dist + r_t_dist) * 0.5

    dist_reward = 1.0 / (1.0 + finger_cabinet_dist**2)
    dist_reward *= dist_reward
    dist_reward = torch.where(
        l_t_dist + r_t_dist <= 0.04,
        dist_reward * 2,
        dist_reward,
    )

    action_penalty = torch.sum(actions**2, dim=-1)

    reward = dist_reward * dist_reward_scale - action_penalty * action_penalty_scale

    resets = torch.where(
        l_t_dist >= 5.5, torch.ones_like(reset_buf), reset_buf
    )  # 5.5 = distance after which hand goes stray
    resets = torch.where(r_t_dist >= 5.5, torch.ones_like(resets), resets)

    successes = torch.where(
        successes == 0,
        torch.where(
            torch.abs(drawer_top_handle_pos[:, 0]) > 1.2,
            torch.ones_like(successes),
            successes,
        ),
        successes,
    )

    resets = torch.where(
        progress_buf >= max_episode_length, torch.ones_like(resets), resets
    )

    cons_successes = torch.where(
        resets > 0, successes * resets, consecutive_successes
    ).mean()

    return reward, resets, progress_buf, successes, cons_successes
