import numpy as np

from roboseq.utils.torch_jit_utils import *
from roboseq.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch # type: ignore
from isaacgym import gymapi # type: ignore

import torch # type: ignore
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

        self.ctr = 0

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        self.num_agents = 1
        self.cfg["env"]["numActions"] = 62 #62

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)  # (2,13)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # (68,2)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)  # (88,13)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)  # (68,)

        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_arnie_dofs + self.num_object_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.arnie_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_arnie_dofs]
        self.arnie_dof_pos = self.arnie_dof_state[..., 0]
        self.arnie_dof_vel = self.arnie_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :,
            self.num_arnie_dofs : self.num_arnie_dofs
            + self.num_object_dofs,
        ]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        self.arnie_default_dof_pos = to_torch( # type: ignore
            [
                -0.9,
                1.57,
                3.12,
                0,
                -0.2,
                0.44,
                -0.9,
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
        ) 

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
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

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
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        arnie_start_pose = gymapi.Transform()
        arnie_start_pose.p = gymapi.Vec3(1.75, 0, 0)
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
        asset_options.vhacd_enabled = False
        asset_options.use_mesh_materials = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1

        arnie_asset = self.gym.load_asset(self.sim, asset_root, arnie_asset_file, asset_options)
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)
        table_asset = self.gym.load_asset(self.sim, asset_root, table_asset_file, asset_options)
        locator_asset = self.gym.create_box(self.sim, 0.05 , 0.05 , 0.05 , asset_options)

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

        self.arnie_dof_lower_limits = to_torch(self.arnie_dof_lower_limits, device=self.device) # type: ignore
        self.arnie_dof_upper_limits = to_torch(self.arnie_dof_upper_limits, device=self.device) # type: ignore
        self.arnie_dof_default_pos = to_torch(self.arnie_dof_default_pos, device=self.device) # type: ignore
        self.arnie_dof_default_vel = to_torch(self.arnie_dof_default_vel, device=self.device) # type: ignore

        self.hand_indices = []
        self.object_indices = []
        self.env_indices = []

        self.object_init_state = []

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(arnie_asset, name)
            for name in self.fingertips
        ]
        
        """Test Code"""
        
        self.cameras = []
        self.camera_tensors = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 256
        self.camera_props.near_plane = 0.1
        self.camera_props.far_plane = 0.2
        self.camera_props.enable_tensors = True

        self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.pointCloudDownsampleNum = 4096
        self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')
        self.u = 64
        self.v = 64
        self.side = 128
        self.camera_v3, self.camera_u3 = torch.meshgrid(torch.arange(0, self.side, device=self.device), torch.arange(0, self.side, device=self.device), indexing='ij')
        
        """Test Code Ends"""

        self.left_hand_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"left_base_link")
        self.left_thumb_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"left_Thumb_Phadist")
        self.left_index_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"left_Index_Phadist")
        self.right_hand_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"right_base_link")
        self.right_thumb_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"right_Thumb_Phadist")
        self.right_index_handle = self.gym.find_asset_rigid_body_index(arnie_asset,"right_Index_Phadist")

        self.object_handle = self.gym.find_asset_rigid_body_index(object_asset,"base")
        
        cam_start_pose = gymapi.Transform()
        cam_start_pose.p = gymapi.Vec3(0.5, 0, 0.5)
        cam_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0,0)

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
            cam_locator = self.gym.create_actor(env_ptr, locator_asset, cam_start_pose, "cam", i, 0, 0)
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
            
            """Test Code"""
            camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
            self.gym.set_camera_location(camera_handle, env_ptr, cam_start_pose.p, object_start_pose.p)  #0.785
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            
            torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
            cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
            cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)

            origin = self.gym.get_env_origin(env_ptr)
            self.env_origin[i][0] = origin.x
            self.env_origin[i][1] = origin.y
            self.env_origin[i][2] = origin.z
            self.camera_tensors.append(torch_cam_tensor)
            self.camera_view_matrixs.append(cam_vinv)
            self.camera_proj_matrixs.append(cam_proj)
            self.cameras.append(camera_handle)
            """Test Code Ends"""

        self.object_init_state = to_torch( # type: ignore
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        
        self.hand_indices = to_torch( # type: ignore
            self.hand_indices, dtype=torch.long, device=self.device
        )
        
        self.object_indices = to_torch( # type: ignore
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
        self.ctr+=1
        
        if self.ctr == 40 : # 8 times in 0ne iteration, once every 5 iterations
            self.render_point_cloud()
            self.ctr = 0

    def compute_full_state(self, asymm_obs=False):
        num_ft_states = 13 * int(self.num_fingertips)  # 130
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 60

        self.obs_buf[:, 0 : self.num_arnie_dofs] = unscale(self.arnie_dof_pos, self.arnie_dof_lower_limits, self.arnie_dof_upper_limits) # type: ignore
        self.obs_buf[:, self.num_arnie_dofs : 2 * self.num_arnie_dofs] = ( self.vel_obs_scale * self.arnie_dof_vel )
        self.obs_buf[:, self.num_arnie_dofs * 2 : 3 * self.num_arnie_dofs] = ( self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_arnie_dofs] )

        fingertip_obs_start = self.num_arnie_dofs * 3  # 192
        
        self.obs_buf[:, fingertip_obs_start : fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)

        hand_pose_start = fingertip_obs_start + num_ft_states

        self.obs_buf[:, hand_pose_start : hand_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_pose_start + 3 : hand_pose_start + 4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1) # type: ignore
        self.obs_buf[:, hand_pose_start + 4 : hand_pose_start + 5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1) # type: ignore
        self.obs_buf[:, hand_pose_start + 5 : hand_pose_start + 6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1) # type: ignore

        self.obs_buf[:, hand_pose_start + 6 : hand_pose_start + 9] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start + 9 : hand_pose_start + 10] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1) # type: ignore
        self.obs_buf[:, hand_pose_start + 10 : hand_pose_start + 11] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1) # type: ignore
        self.obs_buf[:, hand_pose_start + 11 : hand_pose_start + 12] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1) # type: ignore

        action_obs_start = hand_pose_start + 12

        self.obs_buf[:, action_obs_start : action_obs_start + self.num_actions] = self.actions[:, :self.num_actions]

        obj_obs_start = action_obs_start + self.num_actions

        self.obs_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = (self.vel_obs_scale * self.object_angvel)

    def reset(self, env_ids):
        # generate random floats for offsetting
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arnie_dofs + self.num_object_dofs), device=self.device) # type: ignore
        
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
        self.object_dof_pos[env_ids, :] = to_torch([0], device=self.device) # type: ignore
        self.object_dof_vel[env_ids, :] = to_torch([0], device=self.device) # type: ignore
        
        # prev and current targets set to random position for arms    
        self.prev_targets[env_ids, : self.num_arnie_dofs] = pos
        self.cur_targets[env_ids, : self.num_arnie_dofs] = pos
        
        # prev and current targets set to zero for drawer
        self.prev_targets[env_ids, self.num_arnie_dofs : self.num_arnie_dofs + self.num_object_dofs] = to_torch([0], device=self.device) # type: ignore
        self.cur_targets[env_ids, self.num_arnie_dofs : self.num_arnie_dofs + self.num_object_dofs] = to_torch([0], device=self.device) # type: ignore
        
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
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset(env_ids)

        self.actions = actions.clone().to(self.device)
        
        self.cur_targets[:, :self.num_arnie_dofs] = scale(self.actions[:, :self.num_arnie_dofs], self.arnie_dof_lower_limits, self.arnie_dof_upper_limits) # type: ignore
        self.cur_targets[:, :self.num_arnie_dofs] = self.act_moving_average * self.cur_targets[:, :self.num_arnie_dofs] + (1.0 - self.act_moving_average) * self.prev_targets[:, :self.num_arnie_dofs]
        self.cur_targets[:, :self.num_arnie_dofs] = tensor_clamp(self.cur_targets[:, :self.num_arnie_dofs], self.arnie_dof_lower_limits, self.arnie_dof_upper_limits) # type: ignore

        self.prev_targets[:, :self.num_arnie_dofs] = self.cur_targets[:, :self.num_arnie_dofs]
        
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

    def test_point_xyz(self, points, env):
        self.gym.clear_lines(self.viewer)
        
        x = torch.mean(points[:, 0]).item()
        y = torch.mean(points[:, 1]).item()
        z = torch.mean(points[:, 2]).item()

        delta = 0.1
        x_beg = x - delta
        y_beg = y - delta
        z_beg = z - delta

        x_end = x + delta
        y_end = y + delta
        z_end = z + delta

        self.gym.add_lines(self.viewer, env, 1, [x_beg, y, z, x_end, y, z], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [x, y_beg, z, x, y_end, z], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [x, y, z_beg, x, y, z_end], [0.1, 0.1, 0.85])
     
    def rand_row(self, tensor, dim_needed): 
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def image_crop(self, image, u, v, side):  
        dim = image.shape[0]

        u_end = u + side
        v_end = v + side

        u = max(0, min(u, dim - side))
        v = max(0, min(v, dim - side))

        image = image[v:v_end, u:u_end].contiguous()

        return image
        
    def sample_points(self, points, sample_num=1000, sample_method='random'):
        eff_points = points[points[:, 2] > 0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_method == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_method == 'furthest':
            pass
            # TODO Implement Furthest Point Sampling
        return sampled_points

    def render_point_cloud(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        point_clouds = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)

        useSeg = True # use segmentation of object pcd

        for i in range(self.num_envs): # suboptimal to convert between cpu and gpu
            if not useSeg:
                points = depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i], self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, 10, self.device)
            else:
                image = self.image_crop(self.camera_tensors[i], self.u, self.v, self.side)
                points = subset_image_to_point_cloud_GPU(image, self.camera_view_matrixs[i], self.camera_proj_matrixs[i], self.camera_u3, self.camera_v3, self.side, self.side, self.device, 1)
                       
            if points.shape[0] > 0:
                selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_method='random')
                self.test_point_xyz(selected_points, self.env_indices[i])
            else:
                selected_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
            
            point_clouds[i] = selected_points
        
        # Pointcloud visualisation
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_clouds[0, :, :3].cpu().numpy())
        o3d.io.write_point_cloud("pcd.ply", point_cloud)

        self.gym.end_access_image_tensors(self.sim)

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
    l_dist = torch.norm(object_pos - left_ff_pos, p=2, dim=-1) + torch.norm(object_pos - left_th_pos, p=2, dim=-1)

    r_dist = torch.norm(object_pos - right_ff_pos, p=2, dim=-1) + torch.norm(object_pos - right_th_pos, p=2, dim=-1)

    avg_dist = (l_dist + r_dist) * 0.5

    # dist_reward = 1.0 / (1.0 + avg_dist**2)
    # dist_reward *= dist_reward
    # dist_reward = torch.where(
    #     l_dist + r_dist <= 0.04,
    #     dist_reward * 2,
    #     dist_reward,
    # )

    dist_reward = 1.0 / (1.0 + l_dist**2)
    dist_reward *= dist_reward
    dist_reward = torch.where(
        l_dist <= 0.02,
        dist_reward * 2,
        dist_reward
    )
    action_penalty = torch.sum(actions**2, dim=-1)

    reward = dist_reward * dist_reward_scale - action_penalty * action_penalty_scale

    resets = torch.where(
        l_dist >= 5.5, torch.ones_like(reset_buf), reset_buf
    )  # 5.5 = distance after which hand goes stray
    resets = torch.where(r_dist >= 5.5, torch.ones_like(resets), resets)

    successes = torch.where(
        successes == 0,
        torch.where(
            torch.abs(object_pos[:, 2]) > 0.2,
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

@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    depth_buffer = camera_tensor.to(device)

    vinv = camera_view_matrix_inv

    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

@torch.jit.script
def subset_image_to_point_cloud_GPU(tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, device:torch.device, depth_bar:float):
    
    depth_buffer = tensor.to(device)
    
    vinv = camera_view_matrix_inv    
    proj = camera_proj_matrix

    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points
