from unittest import TextTestRunner
from matplotlib.pyplot import axis
from PIL import Image as Im

import numpy as np
import os
import random
import torch
import math

from roboseq.utils.torch_jit_utils import *
from roboseq.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class ArnieGraspCube(BaseTask):
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

        self.is_multi_agent = is_multi_agent

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

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.diana_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        self.object_type = self.cfg["env"]["objectType"]

        self.ignore_z = False

        self.asset_files_dict = {"block": "urdf/objects/cube_multicolor.urdf"}

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get(
                "assetFileNameBlock", self.asset_files_dict["block"]
            )

        self.obs_type = self.cfg["env"]["observationType"]

        self.num_obs_dict = {"full_state": 425}
        self.num_hand_obs = 72 + 95 + 26 + 6
        self.up_axis = "z"

        self.fingertips = ["ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal"]

        self.hand_center = ["palm"]

        self.num_fingertips = len(self.fingertips) * 2

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 31

        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 62

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.left_default_dof_pos = to_torch(
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
                0,
                0,
                0,
                0,
            ],
            dtype=torch.float,
            device=self.device,
        )
        self.right_default_dof_pos = to_torch(
            [
                0.5,
                1.57,
                0,
                0,
                0,
                0,
                3.12,
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

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.left_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_diana_dofs
        ]
        self.left_hand_dof_pos = self.left_hand_dof_state[..., 0]
        self.left_hand_dof_vel = self.left_hand_dof_state[..., 1]

        self.right_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, self.num_diana_dofs : self.num_diana_dofs * 2
        ]
        self.right_hand_dof_pos = self.right_hand_dof_state[..., 0]
        self.right_hand_dof_vel = self.right_hand_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, self.num_diana_dofs * 2 : self.num_diana_dofs * 2 + self.num_object_dofs
        ]
        self.object_dof_pos = self.object_dof_state[..., 0]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

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

        asset_root = "../assets"
        left_hand_asset_file = "urdf/Diana/diana_with_hand.urdf"
        right_hand_asset_file = "urdf/Diana/diana_with_hand.urdf"
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(
            self.sim, table_texture_files
        )

        object_asset_file = "urdf/cube.urdf"

        # load diana hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        left_hand_asset = self.gym.load_asset(
            self.sim, asset_root, left_hand_asset_file, asset_options
        )
        right_hand_asset = self.gym.load_asset(
            self.sim, asset_root, right_hand_asset_file, asset_options
        )

        self.num_diana_bodies = self.gym.get_asset_rigid_body_count(left_hand_asset)
        self.num_diana_shapes = self.gym.get_asset_rigid_shape_count(left_hand_asset)
        self.num_diana_dofs = self.gym.get_asset_dof_count(left_hand_asset)

        # set shadow_hand dof properties
        left_hand_dof_props = self.gym.get_asset_dof_properties(left_hand_asset)
        right_hand_dof_props = self.gym.get_asset_dof_properties(right_hand_asset)

        self.diana_dof_lower_limits = []
        self.diana_dof_upper_limits = []
        self.diana_dof_default_pos = []
        self.diana_dof_default_vel = []

        for i in range(self.num_diana_dofs):
            self.diana_dof_lower_limits.append(left_hand_dof_props["lower"][i])
            self.diana_dof_upper_limits.append(left_hand_dof_props["upper"][i])
            self.diana_dof_default_pos.append(0.0)
            self.diana_dof_default_vel.append(0.0)

        self.diana_dof_lower_limits = to_torch(
            self.diana_dof_lower_limits, device=self.device
        )
        self.diana_dof_upper_limits = to_torch(
            self.diana_dof_upper_limits, device=self.device
        )
        self.diana_dof_default_pos = to_torch(
            self.diana_dof_default_pos, device=self.device
        )
        self.diana_dof_default_vel = to_torch(
            self.diana_dof_default_vel, device=self.device
        )

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 100000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        object_asset = self.gym.load_asset(
            self.sim, asset_root, object_asset_file, object_asset_options
        )
        block_asset_file = "urdf/objects/cube_multicolor1.urdf"
        block_asset = self.gym.load_asset(
            self.sim, asset_root, block_asset_file, object_asset_options
        )

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(
            self.sim, asset_root, object_asset_file, object_asset_options
        )

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

        # set object dof properties
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)

        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props["lower"][i])
            self.object_dof_upper_limits.append(object_dof_props["upper"][i])

        self.object_dof_lower_limits = to_torch(
            self.object_dof_lower_limits, device=self.device
        )
        self.object_dof_upper_limits = to_torch(
            self.object_dof_upper_limits, device=self.device
        )

        # create table asset
        table_dims = gymapi.Vec3(1.0, 1.0, 0.6)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions()
        )

        left_hand_start_pose = gymapi.Transform()
        left_hand_start_pose.p = gymapi.Vec3(-1.25, 0.1, 1.45)
        left_hand_start_pose.r = gymapi.Quat().from_euler_zyx(
            -1.5652925671162337, 0, -0.227
        )

        right_hand_start_pose = gymapi.Transform()
        right_hand_start_pose.p = gymapi.Vec3(-1.25, -0.1, 1.45)
        right_hand_start_pose.r = gymapi.Quat().from_euler_zyx(
            1.5652925671162337, 0, 0.227
        )

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, -0.2, 0.65)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        pose_dx, pose_dy, pose_dz = 1.0, 0.0, -0.0

        self.goal_displacement = gymapi.Vec3(-0.0, 0.0, 1)
        self.goal_displacement_tensor = to_torch(
            [
                self.goal_displacement.x,
                self.goal_displacement.y,
                self.goal_displacement.z,
            ],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0.0, 0, 0)

        # compute aggregate size
        max_agg_bodies = self.num_diana_bodies * 2 + 3 * self.num_object_bodies + 1
        max_agg_shapes = self.num_diana_shapes * 2 + 3 * self.num_object_shapes + 1

        self.dianas = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.left_hand_indices = []
        self.right_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.left_fingertip_handles = [
            self.gym.find_asset_rigid_body_index(left_hand_asset, name)
            for name in self.fingertips
        ]
        self.right_fingertip_handles = [
            self.gym.find_asset_rigid_body_index(right_hand_asset, name)
            for name in self.fingertips
        ]

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            left_hand_actor = self.gym.create_actor(
                env_ptr, left_hand_asset, left_hand_start_pose, "left_hand", i, -1, 0
            )
            right_hand_actor = self.gym.create_actor(
                env_ptr, right_hand_asset, right_hand_start_pose, "right_hand", i, -1, 0
            )

            self.hand_start_states.append(
                [
                    left_hand_start_pose.p.x,
                    left_hand_start_pose.p.y,
                    left_hand_start_pose.p.z,
                    left_hand_start_pose.r.x,
                    left_hand_start_pose.r.y,
                    left_hand_start_pose.r.z,
                    left_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            self.gym.set_actor_dof_properties(
                env_ptr, left_hand_actor, left_hand_dof_props
            )
            left_hand_idx = self.gym.get_actor_index(
                env_ptr, left_hand_actor, gymapi.DOMAIN_SIM
            )
            self.left_hand_indices.append(left_hand_idx)

            self.gym.set_actor_dof_properties(
                env_ptr, right_hand_actor, right_hand_dof_props
            )
            right_hand_idx = self.gym.get_actor_index(
                env_ptr, right_hand_actor, gymapi.DOMAIN_SIM
            )
            self.right_hand_indices.append(right_hand_idx)

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, left_hand_actor)

            # add object
            object_handle = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "object", i, 0, 0
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
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(
                env_ptr,
                goal_asset,
                goal_start_pose,
                "goal_object",
                i + self.num_envs,
                0,
                0,
            )
            goal_object_idx = self.gym.get_actor_index(
                env_ptr, goal_handle, gymapi.DOMAIN_SIM
            )
            self.goal_object_indices.append(goal_object_idx)

            # add table
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, -1, 0
            )
            self.gym.set_rigid_body_texture(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle
            )
            table_idx = self.gym.get_actor_index(
                env_ptr, table_handle, gymapi.DOMAIN_SIM
            )
            self.table_indices.append(table_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr,
                    object_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )
                self.gym.set_rigid_body_color(
                    env_ptr,
                    goal_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.dianas.append(left_hand_actor)

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()

        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(
            self.hand_start_states, device=self.device
        ).view(self.num_envs, 13)

        self.left_fingertip_handles = to_torch(
            self.left_fingertip_handles, dtype=torch.long, device=self.device
        )
        self.right_fingertip_handles = to_torch(
            self.right_fingertip_handles, dtype=torch.long, device=self.device
        )

        self.left_hand_indices = to_torch(
            self.left_hand_indices, dtype=torch.long, device=self.device
        )
        self.right_hand_indices = to_torch(
            self.right_hand_indices, dtype=torch.long, device=self.device
        )

        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        self.goal_object_indices = to_torch(
            self.goal_object_indices, dtype=torch.long, device=self.device
        )
        self.table_indices = to_torch(
            self.table_indices, dtype=torch.long, device=self.device
        )

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_hand_reward_lift_block(
            self.rew_buf,
            self.reset_buf,
            self.reset_goal_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.goal_pos,
            self.goal_rot,
            self.left_hand_pos,
            self.right_hand_pos,
            self.right_hand_ff_pos,
            self.right_hand_mf_pos,
            self.right_hand_rf_pos,
            self.right_hand_lf_pos,
            self.right_hand_th_pos,
            self.left_hand_ff_pos,
            self.left_hand_mf_pos,
            self.left_hand_rf_pos,
            self.left_hand_lf_pos,
            self.left_hand_th_pos,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.rot_eps,
            self.actions,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
            False,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        # self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        # self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.object_pos = self.rigid_body_states[:, 64, 0:3]
        self.object_rot = self.rigid_body_states[:, 64, 3:7]
        self.object_pos = self.object_pos + quat_apply(
            self.object_rot,
            to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.0,
        )
        self.object_pos = self.object_pos + quat_apply(
            self.object_rot,
            to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.0,
        )
        self.object_pos = self.object_pos + quat_apply(
            self.object_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0,
        )

        self.goal_pos = self.rigid_body_states[:, 65, 0:3]
        self.goal_rot = self.rigid_body_states[:, 65, 3:7]
        self.goal_pos = self.goal_pos + quat_apply(
            self.goal_rot,
            to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.0,
        )
        self.goal_pos = self.goal_pos + quat_apply(
            self.goal_rot,
            to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.0,
        )
        self.goal_pos = self.goal_pos + quat_apply(
            self.goal_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0,
        )

        self.left_hand_pos = self.rigid_body_states[:, 7, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, 7, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(
            self.left_hand_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08,
        )
        self.left_hand_pos = self.left_hand_pos + quat_apply(
            self.left_hand_rot,
            to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02,
        )

        self.right_hand_pos = self.rigid_body_states[:, 39, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 39, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(
            self.right_hand_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08,
        )
        self.right_hand_pos = self.right_hand_pos + quat_apply(
            self.right_hand_rot,
            to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02,
        )

        # right hand finger
        self.right_hand_ff_pos = self.rigid_body_states[:, 45, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, 45, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(
            self.right_hand_ff_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )
        self.right_hand_mf_pos = self.rigid_body_states[:, 54, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, 54, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(
            self.right_hand_mf_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )
        self.right_hand_rf_pos = self.rigid_body_states[:, 58, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, 58, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(
            self.right_hand_rf_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )
        self.right_hand_lf_pos = self.rigid_body_states[:, 50, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, 50, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(
            self.right_hand_lf_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )
        self.right_hand_th_pos = self.rigid_body_states[:, 63, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, 63, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(
            self.right_hand_th_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )

        self.left_hand_ff_pos = self.rigid_body_states[:, 13, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, 13, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(
            self.left_hand_ff_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )
        self.left_hand_mf_pos = self.rigid_body_states[:, 22, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, 22, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(
            self.left_hand_mf_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )
        self.left_hand_rf_pos = self.rigid_body_states[:, 26, 0:3]
        self.left_hand_rf_rot = self.rigid_body_states[:, 26, 3:7]
        self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(
            self.left_hand_rf_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )
        self.left_hand_lf_pos = self.rigid_body_states[:, 18, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, 18, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(
            self.left_hand_lf_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )
        self.left_hand_th_pos = self.rigid_body_states[:, 31, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, 31, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(
            self.left_hand_th_rot,
            to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02,
        )

        # self.goal_pos = to_torch([0, 0.2, 5.6], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)) # -0.3, 0, 0.6
        # self.goal_rot = self.goal_states[:, 3:7]

        self.left_fingertip_state = self.rigid_body_states[
            :, self.left_fingertip_handles
        ][:, :, 0:13]
        self.left_fingertip_pos = self.rigid_body_states[
            :, self.left_fingertip_handles
        ][:, :, 0:3]
        self.right_fingertip_state = self.rigid_body_states[
            :, self.right_fingertip_handles
        ][:, :, 0:13]
        self.right_fingertip_pos = self.rigid_body_states[
            :, self.right_fingertip_handles
        ][:, :, 0:3]

        self.compute_full_state()

    def compute_full_state(self, asymm_obs=False):
        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0 : self.num_diana_dofs] = unscale(
            self.left_hand_dof_pos,
            self.diana_dof_lower_limits,
            self.diana_dof_upper_limits,
        )
        self.obs_buf[:, self.num_diana_dofs : 2 * self.num_diana_dofs] = (
            self.vel_obs_scale * self.left_hand_dof_vel
        )

        left_fingertip_obs_start = 62
        self.obs_buf[
            :, left_fingertip_obs_start : left_fingertip_obs_start + num_ft_states
        ] = self.left_fingertip_state.reshape(self.num_envs, num_ft_states)

        left_hand_pose_start = left_fingertip_obs_start + 65
        self.obs_buf[
            :, left_hand_pose_start : left_hand_pose_start + 3
        ] = self.left_hand_pos
        self.obs_buf[
            :, left_hand_pose_start + 3 : left_hand_pose_start + 4
        ] = get_euler_xyz(self.hand_orientations[self.left_hand_indices, :])[
            0
        ].unsqueeze(
            -1
        )
        self.obs_buf[
            :, left_hand_pose_start + 4 : left_hand_pose_start + 5
        ] = get_euler_xyz(self.hand_orientations[self.left_hand_indices, :])[
            1
        ].unsqueeze(
            -1
        )
        self.obs_buf[
            :, left_hand_pose_start + 5 : left_hand_pose_start + 6
        ] = get_euler_xyz(self.hand_orientations[self.left_hand_indices, :])[
            2
        ].unsqueeze(
            -1
        )

        left_action_obs_start = left_hand_pose_start + 6
        self.obs_buf[
            :, left_action_obs_start : left_action_obs_start + 31
        ] = self.actions[:, :31]

        right_hand_start = left_action_obs_start + 31
        self.obs_buf[
            :, right_hand_start : self.num_diana_dofs + right_hand_start
        ] = unscale(
            self.right_hand_dof_pos,
            self.diana_dof_lower_limits,
            self.diana_dof_upper_limits,
        )
        self.obs_buf[
            :,
            self.num_diana_dofs
            + right_hand_start : 2 * self.num_diana_dofs
            + right_hand_start,
        ] = (
            self.vel_obs_scale * self.right_hand_dof_vel
        )

        right_fingertip_obs_start = right_hand_start + 62
        self.obs_buf[
            :, right_fingertip_obs_start : right_fingertip_obs_start + num_ft_states
        ] = self.right_fingertip_state.reshape(self.num_envs, num_ft_states)

        right_hand_pose_start = right_fingertip_obs_start + 65
        self.obs_buf[
            :, right_hand_pose_start : right_hand_pose_start + 3
        ] = self.right_hand_pos
        self.obs_buf[
            :, right_hand_pose_start + 3 : right_hand_pose_start + 4
        ] = get_euler_xyz(self.hand_orientations[self.right_hand_indices, :])[
            0
        ].unsqueeze(
            -1
        )
        self.obs_buf[
            :, right_hand_pose_start + 4 : right_hand_pose_start + 5
        ] = get_euler_xyz(self.hand_orientations[self.right_hand_indices, :])[
            1
        ].unsqueeze(
            -1
        )
        self.obs_buf[
            :, right_hand_pose_start + 5 : right_hand_pose_start + 6
        ] = get_euler_xyz(self.hand_orientations[self.right_hand_indices, :])[
            2
        ].unsqueeze(
            -1
        )

        right_action_obs_start = right_hand_pose_start + 6
        self.obs_buf[
            :, right_action_obs_start : right_action_obs_start + 31
        ] = self.actions[:, 31:]

        obj_obs_start = right_action_obs_start + 31
        self.obs_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = (
            self.vel_obs_scale * self.object_angvel
        )
        self.obs_buf[:, obj_obs_start + 13 : obj_obs_start + 16] = self.object_pos
        self.obs_buf[:, obj_obs_start + 16 : obj_obs_start + 20] = self.object_rot
        self.obs_buf[:, obj_obs_start + 20 : obj_obs_start + 23] = self.goal_pos
        self.obs_buf[:, obj_obs_start + 23 : obj_obs_start + 27] = self.goal_rot

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 2] += 10.0

        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = (
            self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        )
        self.root_state_tensor[
            self.goal_object_indices[env_ids], 3:7
        ] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[
            self.goal_object_indices[env_ids], 7:13
        ] = torch.zeros_like(
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
        )

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices),
                len(env_ids),
            )
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_diana_dofs * 2 + 5), device=self.device
        )

        self.reset_target_pose(env_ids)

        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[
            env_ids
        ].clone()

        new_object_rot = randomize_rotation(
            rand_floats[:, 3],
            rand_floats[:, 4],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )

        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )

        object_indices = torch.unique(
            torch.cat(
                [
                    self.object_indices[env_ids],
                    self.goal_object_indices[env_ids],
                    self.goal_object_indices[goal_env_ids],
                ]
            ).to(torch.int32)
        )

        delta_max = self.diana_dof_upper_limits - self.diana_dof_default_pos
        delta_min = self.diana_dof_lower_limits - self.diana_dof_default_pos
        rand_delta = (
            delta_min
            + (delta_max - delta_min) * rand_floats[:, 5 : 5 + self.num_diana_dofs]
        )

        left_pos = self.left_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        right_pos = self.right_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.left_hand_dof_pos[env_ids, :] = left_pos
        self.right_hand_dof_pos[env_ids, :] = right_pos

        self.left_hand_dof_vel[env_ids, :] = (
            self.diana_dof_default_vel
            + self.reset_dof_vel_noise
            * rand_floats[:, 5 + self.num_diana_dofs : 5 + self.num_diana_dofs * 2]
        )
        self.right_hand_dof_vel[env_ids, :] = (
            self.diana_dof_default_vel
            + self.reset_dof_vel_noise
            * rand_floats[:, 5 + self.num_diana_dofs : 5 + self.num_diana_dofs * 2]
        )

        self.prev_targets[env_ids, : self.num_diana_dofs] = left_pos
        self.cur_targets[env_ids, : self.num_diana_dofs] = left_pos

        self.prev_targets[
            env_ids, self.num_diana_dofs : self.num_diana_dofs * 2
        ] = right_pos
        self.cur_targets[
            env_ids, self.num_diana_dofs : self.num_diana_dofs * 2
        ] = right_pos

        left_hand_indices = self.left_hand_indices[env_ids].to(torch.int32)
        right_hand_indices = self.right_hand_indices[env_ids].to(torch.int32)

        all_hand_indices = torch.unique(
            torch.cat([left_hand_indices, right_hand_indices]).to(torch.int32)
        )

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets)
        )

        all_indices = torch.unique(
            torch.cat(
                [
                    all_hand_indices,
                    self.object_indices[env_ids],
                    self.table_indices[env_ids],
                ]
            ).to(torch.int32)
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices),
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        dof_low = self.diana_dof_lower_limits.unsqueeze(0).repeat(1, 2)
        dof_up = self.diana_dof_upper_limits.unsqueeze(0).repeat(1, 2)

        self.cur_targets[:, 31:] = scale(
            self.actions[:, 31:], dof_low[:, 31:], dof_up[:, 31:]
        )
        self.cur_targets[:, 31:] = (
            self.act_moving_average * self.cur_targets[:, 31:]
            + (1.0 - self.act_moving_average) * self.prev_targets[:, 31:]
        )
        self.cur_targets[:, 31:] = tensor_clamp(
            self.cur_targets[:, 31:], dof_low[:, 31:], dof_up[:, 31:]
        )

        self.prev_targets[:, 31:] = self.cur_targets[:, 31:]

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.compute_observations()
        self.compute_reward(self.actions)

@torch.jit.script
def compute_hand_reward_lift_block(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    left_hand_pos,
    right_hand_pos,
    right_hand_ff_pos,
    right_hand_mf_pos,
    right_hand_rf_pos,
    right_hand_lf_pos,
    right_hand_th_pos,
    left_hand_ff_pos,
    left_hand_mf_pos,
    left_hand_rf_pos,
    left_hand_lf_pos,
    left_hand_th_pos,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
    ignore_z_rot: bool,
):
    right_goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

    right_hand_dist = torch.norm(object_pos - right_hand_pos, p=2, dim=-1)

    right_hand_finger_dist = (
        torch.norm(object_pos - right_hand_ff_pos, p=2, dim=-1)
        + torch.norm(object_pos - right_hand_mf_pos, p=2, dim=-1)
        + torch.norm(object_pos - right_hand_rf_pos, p=2, dim=-1)
        + torch.norm(object_pos - right_hand_lf_pos, p=2, dim=-1)
        + torch.norm(object_pos - right_hand_th_pos, p=2, dim=-1)
    )

    right_hand_dist_rew = torch.exp(-10 * right_hand_dist)
    right_hand_fingertip_dist_rew = torch.exp(-1 * right_hand_finger_dist)
    right_goal_dist_rew = torch.exp(-1 * right_goal_dist)

    action_penalty = torch.sum(actions**2, dim=-1)

    up_rew = torch.zeros_like(right_hand_dist_rew)
    up_rew = torch.exp(-10 * torch.norm(target_pos - object_pos, p=2, dim=-1)) * 2

    # reward = (right_hand_fingertip_dist_rew + right_hand_dist_rew + right_goal_dist ) * dist_reward_scale - action_penalty * action_penalty_scale
    reward = (
        right_hand_dist_rew + right_goal_dist_rew
    ) * dist_reward_scale - action_penalty * action_penalty_scale
    resets = torch.where(
        right_hand_dist_rew <= 0, torch.ones_like(reset_buf), reset_buf
    )
    resets = torch.where(right_hand_finger_dist >= 2, torch.ones_like(resets), resets)
    # resets = torch.where(left_hand_finger_dist >= 2, torch.ones_like(resets), resets)

    successes = torch.where(
        successes == 0,
        torch.where(
            torch.norm(target_pos - object_pos, p=2, dim=-1) > 0.5,
            torch.ones_like(successes),
            successes,
        ),
        successes,
    )

    resets = torch.where(
        progress_buf >= max_episode_length, torch.ones_like(resets), resets
    )

    goal_resets = torch.zeros_like(resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        resets > 0, successes * resets, consecutive_successes
    ).mean()

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )
