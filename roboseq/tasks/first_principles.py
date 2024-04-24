import math
import numpy as np
import torch

from roboseq.utils.torch_jit_utils import *
from roboseq.tasks.hand_base.base_task import BaseTask

from isaacgym import gymtorch
from isaacgym import gymapi

def iprint(*strings):
    print(strings)
    exit()

class FirstPrinciples(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # config init
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        
        # randomisation init
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.action_scale = 7.5

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.arm_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        # sim init
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        self.up_axis = 'z'

        self.cfg["env"]["numObservations"] = 40
        self.cfg["env"]["numStates"] = 14
        self.cfg["env"]["numActions"] = 7

        super().__init__(cfg=self.cfg)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        self.arm_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_dofs]
        self.arm_dof_pos = self.arm_dof_state[..., 0]
        self.arm_dof_vel = self.arm_dof_state[..., 1]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.arm_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        arm_asset_file = "urdf/Diana/diana_v2.urdf"
        cube_asset_file = "urdf/cube.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        arm_asset = self.gym.load_asset(self.sim, asset_root, arm_asset_file, asset_options)

        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        
        cube_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, asset_options)

        table_asset = self.gym.create_box(self.sim, 0.4 , 0.4 , 0.4 , gymapi.AssetOptions())

        arm_start_pose = gymapi.Transform()
        arm_start_pose.p = gymapi.Vec3(0.5, -0.2, 0)
        arm_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        cube_start_pose = gymapi.Transform()
        cube_start_pose.p = gymapi.Vec3(0, 0, 0.4)
        cube_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0, 0, 0.2)
        table_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        ## get asset properties for later use

        # Arm
        self.num_arm_bodies = self.gym.get_asset_rigid_body_count(arm_asset)
        self.num_arm_shapes = self.gym.get_asset_rigid_shape_count(arm_asset)
        self.num_arm_dofs = self.gym.get_asset_dof_count(arm_asset)
        arm_dof_props = self.gym.get_asset_dof_properties(arm_asset)

        self.arm_dof_lower_limits = []
        self.arm_dof_upper_limits = []
        self.arm_dof_default_pos = []
        self.arm_dof_default_vel = []

        for i in range(self.num_arm_dofs):
            self.arm_dof_lower_limits.append(arm_dof_props['lower'][i])
            self.arm_dof_upper_limits.append(arm_dof_props['upper'][i])
            self.arm_dof_default_pos.append(0.0)
            self.arm_dof_default_vel.append(0.0)
        
        self.arm_dof_lower_limits = to_torch(self.arm_dof_lower_limits, device=self.device)
        self.arm_dof_upper_limits = to_torch(self.arm_dof_upper_limits, device=self.device)
        self.arm_dof_default_pos = to_torch(self.arm_dof_default_pos, device=self.device)
        self.arm_dof_default_vel = to_torch(self.arm_dof_default_vel, device=self.device)

        # Cube
        self.num_cube_bodies = self.gym.get_asset_rigid_body_count(cube_asset)
        self.num_cube_shapes = self.gym.get_asset_rigid_shape_count(cube_asset)
        self.num_cube_dofs = self.gym.get_asset_dof_count(cube_asset)

        self.cube_dof_lower_limits = []
        self.cube_dof_upper_limits = []

        self.end_effector_handle = self.gym.find_asset_rigid_body_index(arm_asset,"link_7")

        # quick access tensors
        self.arms = []
        self.envs = []

        self.cube_init_state = []
        self.arm_init_state = []

        self.arm_indices = []
        self.cube_indices = []
        self.table_indices = []

        # Env Creation Loop
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            arm_actor = self.gym.create_actor(env_ptr, arm_asset, arm_start_pose, "arm", i, 0, 0)
            cube_actor = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", i, 0, 0)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)

            self.arm_init_state.append([arm_start_pose.p.x, arm_start_pose.p.y, arm_start_pose.p.z,
                                           arm_start_pose.r.x, arm_start_pose.r.y, arm_start_pose.r.z, arm_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, arm_actor, arm_dof_props)
            arm_idx = self.gym.get_actor_index(env_ptr, arm_actor, gymapi.DOMAIN_SIM)
            self.arm_indices.append(arm_idx)

            self.cube_init_state.append([cube_start_pose.p.x, cube_start_pose.p.y, cube_start_pose.p.z,
                                           cube_start_pose.r.x, cube_start_pose.r.y, cube_start_pose.r.z, cube_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            cube_idx = self.gym.get_actor_index(env_ptr, cube_actor, gymapi.DOMAIN_SIM)
            self.cube_indices.append(cube_idx)

            table_idx = self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            self.envs.append(env_ptr)
            self.arms.append(arm_actor)

        self.arm_init_state = to_torch(self.arm_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.cube_init_state = to_torch(self.cube_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = \
        compute_hand_reward(
            self.reset_buf, self.progress_buf, 
            self.successes, self.consecutive_successes, self.max_episode_length, 
            self.cube_pos,  
            self.end_effector_pos,  
            self.dist_reward_scale,  
            self.actions, self.action_penalty_scale
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.cube_pose = self.root_state_tensor[self.cube_indices, 0:7]
        self.cube_pos = self.root_state_tensor[self.cube_indices, 0:3]
        self.cube_rot = self.root_state_tensor[self.cube_indices, 3:7]
        self.cube_linvel = self.root_state_tensor[self.cube_indices, 7:10]
        self.cube_angvel = self.root_state_tensor[self.cube_indices, 10:13]

        self.end_effector_pose = self.rigid_body_states[:, self.end_effector_handle, 0:7]
        self.end_effector_pos = self.rigid_body_states[:, self.end_effector_handle, 0:3] 
        self.end_effector_rot = self.rigid_body_states[:, self.end_effector_handle, 3:7]
        self.end_effector_linvel = self.rigid_body_states[:, self.end_effector_handle, 7:10] 
        self.end_effector_angvel = self.rigid_body_states[:, self.end_effector_handle, 10:13]
        
        self.compute_full_state()

    def compute_full_state(self):
        # Scaled DoF Positions
        self.obs_buf[:, 0 : self.num_arm_dofs] = unscale(self.arm_dof_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits)
        
        action_obs_start = self.num_arm_dofs
        
        # Actions
        self.obs_buf[:, action_obs_start : action_obs_start + self.num_actions] = self.actions[:, :self.num_actions]

        cube_obs_start = action_obs_start + self.num_actions
        
        # Cube Position, Lin and Ang Vel
        self.obs_buf[:, cube_obs_start : cube_obs_start + 7] = self.cube_pose
        self.obs_buf[:, cube_obs_start + 7 : cube_obs_start + 10] = self.cube_linvel
        self.obs_buf[:, cube_obs_start + 10 : cube_obs_start + 13] = (self.vel_obs_scale * self.cube_angvel)

        end_effector_obs_start = cube_obs_start + 13

        # End Effector Position, Lin and Ang Vel
        self.obs_buf[:, end_effector_obs_start : end_effector_obs_start + 7] = self.end_effector_pose
        self.obs_buf[:, end_effector_obs_start + 7 : end_effector_obs_start + 10] = self.end_effector_linvel
        self.obs_buf[:, end_effector_obs_start + 10 : end_effector_obs_start + 13] = (self.vel_obs_scale * self.end_effector_angvel)

    def reset(self, env_ids):
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_dofs ), device=self.device)

        # reset cube
        self.root_state_tensor[self.cube_indices[env_ids]] = self.cube_init_state[env_ids].clone()
        self.root_state_tensor[self.cube_indices[env_ids], 0:2] = self.cube_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.cube_indices[env_ids], self.up_axis_idx] = self.cube_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        self.root_state_tensor[self.cube_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.cube_indices[env_ids], 7:13])

        delta_max = self.arm_dof_upper_limits - self.arm_dof_default_pos
        delta_min = self.arm_dof_lower_limits - self.arm_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_arm_dofs]

        pos = self.arm_dof_default_pos + self.reset_dof_pos_noise * rand_delta

        self.arm_dof_pos[env_ids, :] = pos

        self.arm_dof_vel[env_ids, :] = self.arm_dof_default_vel + self.reset_dof_vel_noise * rand_floats[:, :self.num_arm_dofs]   

        self.prev_targets[env_ids, :self.num_arm_dofs] = pos
        self.cur_targets[env_ids, :self.num_arm_dofs] = pos

        arm_indices = self.arm_indices[env_ids].to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(arm_indices), len(arm_indices))  

    
        all_indices = torch.unique(torch.cat([arm_indices,
                                              self.cube_indices[env_ids],
                                              self.table_indices[env_ids]]).to(torch.int32))


        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(arm_indices), len(arm_indices))
                                              
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        
        self.target_pos = self.end_effector_pos.clone()
        self.target_pos += self.actions * self.dt * self.action_scale
        
        target_rot = gymapi.Quat.from_euler_zyx(-0.5 * math.pi, 0, 0)  # x,y,z
        self.target_rot = torch.tensor([[target_rot.x, target_rot.y, target_rot.z, target_rot.w]] * self.num_envs, dtype=torch.float32).to(self.device)

        pos_err = self.target_pos - self.end_effector_pos
        orn_err = orientation_error(self.target_rot, self.end_effector_rot)

        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)  # unsqueeze(-1) meaning up one dim internal
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        damping = 0.05
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)      ###########17
        
        self.arm_dof_targets[:, :7] = self.arm_dof_pos[:, :7] + u.squeeze(-1) 

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.arm_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

@torch.jit.script
def compute_hand_reward(
    reset_buf, progress_buf, successes, consecutive_successes, max_episode_length: float, 
    cube_pos, 
    end_effector_pos, 
    dist_reward_scale: float,
    actions, action_penalty_scale: float
):
    dist = torch.norm(end_effector_pos - cube_pos, p=2, dim=-1)
    
    dist_reward = 1.0 / (1.0 + dist**2)
    dist_reward *= dist_reward
    dist_reward = torch.where(
        dist <= 0.02,
        dist_reward * 2,
        dist_reward
    )

    action_penalty = torch.sum(actions ** 2, dim=-1)

    reward = dist_reward * dist_reward_scale - action_penalty * action_penalty_scale

    resets = torch.where(dist >= 1.5, torch.ones_like(reset_buf), reset_buf)
   
    successes = torch.where(successes == 0, torch.where(dist < 0.01, torch.ones_like(successes), successes), successes)

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes).mean()

    return reward, resets, progress_buf, successes, cons_successes

    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot