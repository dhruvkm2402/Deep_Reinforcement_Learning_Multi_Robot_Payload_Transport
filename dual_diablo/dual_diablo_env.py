from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.managers import SceneEntityCfg
import numpy as np
import random
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
# from isaacsim.util.debug_draw import _debug_draw       # This is useful for debugging. It displays waypoints (will work only when not running in headless mode)
# draw = _debug_draw.acquire_debug_draw_interface()
import math
##
# Pre-defined configs
##
from isaaclab_assets.robots.dual_diablo import DUAL_DIABLO_CFG, DUAL_DIABLO_CFG2   # isort: skip   # Change this

coordinates = np.loadtxt('your path to file/GradualSine.csv', delimiter=',', skiprows=1) # Waypoints for training

x_coords = coordinates[:, 0]
y_coords = coordinates[:, 1] 
crosstrackerror = []
headingangle_error = []
boxpose = []
diablo1_pose = []
diablo2_pose = []
act=[]

coordinates = np.stack((x_coords, y_coords), axis=-1)
@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=500,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 0.9),
            "dynamic_friction_range": (0.4, 0.7),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=500,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.5, 2.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )



@configclass
class DualDiabloEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(
    dt=1 / 200,
    render_interval=4,
    use_fabric=True,
    enable_scene_query_support=False,
    gravity=(0.0, 0.0, -9.81),

    physics_material=RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0
    ),
    
    physx=PhysxCfg(
        solver_type=1,
        max_position_iteration_count=4,
        max_velocity_iteration_count=0,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.04,
        friction_correlation_distance=0.025,
        enable_stabilization=True,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=5 * 2**15,
        gpu_found_lost_pairs_capacity=2**21,
        gpu_found_lost_aggregate_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**21,
        gpu_heap_capacity=2**26,
        gpu_temp_buffer_capacity=2**24,
        gpu_max_num_partitions=8,
        gpu_max_soft_body_contacts=2**20,
        gpu_max_particle_contacts=2**20,
    )
)
    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # robot 
    robot: ArticulationCfg = DUAL_DIABLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot2: ArticulationCfg = DUAL_DIABLO_CFG2.replace(prim_path="/World/envs/env_.*/Robot2")
    leftwheel_dof_name_ego = "left_j3"
    rightwheel_dof_name_ego = "right_j3"

    #scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    #env

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"Your Path to the usd file",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=0.1,
            ),
            scale = (1.75,0.14,0.06), # Modify it to reflect the size of payload available (Original size 1 x 1 x 1 m)
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.57), rot=(1.0,0.0,0.0,0.0)),
    )
    
    decimation = 4 # Sets the frequency to 50 Hz decimation*sim_dt
    episode_length_s = 60
    action_scale = 1  # [N]
    action_space = 5
    observation_space = 17
    state_space = 0

class DualDiabloEnv(DirectRLEnv):
    cfg: DualDiabloEnvCfg

    def __init__(self, cfg: DualDiabloEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._leftwheel_dof_idx, _ = self.diablo.find_joints(self.cfg.leftwheel_dof_name_ego)  
        self._rightwheel_dof_idx, _ = self.diablo.find_joints(self.cfg.rightwheel_dof_name_ego)

        self._leftj1_dof_idx, _ = self.diablo.find_joints("left_j1") # Hip joint                           
        
        self._leftj2_dof_idx, _ = self.diablo.find_joints("left_j2") # Knee joint

        self._rightj1_dof_idx, _ = self.diablo.find_joints("right_j1")                            
        
        self._rightj2_dof_idx, _ = self.diablo.find_joints("right_j2")


        self._leftwheel_dof_idx_f, _ = self.diablo2.find_joints(self.cfg.leftwheel_dof_name_ego)                      
        
        self._rightwheel_dof_idx_f, _ = self.diablo2.find_joints(self.cfg.rightwheel_dof_name_ego)
        
        #self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.diablo.data.joint_pos
        self.joint_vel = self.diablo.data.joint_vel

        self.joint_pos2 = self.diablo2.data.joint_pos
        self.joint_vel2 = self.diablo2.data.joint_vel


    def _setup_scene(self):
        self.diablo = Articulation(self.cfg.robot)
        self.diablo2 = Articulation(self.cfg.robot2)
        self.object = RigidObject(self.cfg.object_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["diablo"] = self.diablo
        self.scene.articulations["diablo2"] = self.diablo2
        self.scene.rigid_objects["object"] = self.object

        #Generated offseted paths
        self._num_per_row = int(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / self._num_per_row)
        num_cols = np.ceil(self.num_envs / num_rows)
        env_spacing = 5.0

        row_offset = 0.5 * env_spacing * (num_rows - 1)
        col_offset = 0.5 * env_spacing * (num_cols - 1)
        
        coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)

        translations = []
        

        for i in range(self.num_envs):      # This is how isaac sim spawns each parallel environment. Referenced from isaac sim's grid based approach
            # compute transform
            row = i // num_cols
            col = i % num_cols
            x = row_offset - row * env_spacing
            y = col * env_spacing - col_offset
            translations.append([x,y])
        translations_array = np.array(translations)
        translations_tensor = torch.tensor(translations_array, dtype=torch.float32)
        
        self.translated_coordinates = coordinates_tensor.unsqueeze(0) + translations_tensor.unsqueeze(1) # Apply translations for each parallel environment

        ## Uncomment for displaying waypoints
        # translated_coordinates_numpy = self.translated_coordinates.numpy()
        # num_envs = translated_coordinates_numpy.shape[0]
        # num_points = translated_coordinates_numpy.shape[1]
        # colors = [(random.uniform(1.0, 1.0), random.uniform(1.0, 1.0), random.uniform(1.0, 1.0), 1) for _ in range(num_points)]
        # sizes = [5 for _ in range(num_points)]
        
        #Loop through each environment and prepare the points for drawing
        # for env_index in range(num_envs):
        #     point_list = []
            
        #     for point_index in range(num_points):
        #         # Extract x, y, and assume z as 0 for 2D points; modify if you have a z component
        #         x, y = translated_coordinates_numpy[env_index, point_index]
        #         z = 0.1  # Set z to 0, or you can include it if you have a 3D point

        #         point_list.append((x, y, z))

        # #    # Draw the path for the current environment
        #     draw.draw_points(point_list, colors, sizes)
        
        #The result is a tensor of shape (num_envs, num_points, 2)
        # print(translated_coordinates.size())
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions[:,0] = torch.clamp(actions[:,0].clone(), min=-0.5, max=0.5) # Body twist of ego robot (linear velocity)
        
        self.actions[:,1] = torch.clamp(actions[:,1].clone(), min=-1.0, max=1.0) # Body twist of ego robot (angular velocity)
        
        self.actions[:,2] = torch.clamp(actions[:,2].clone(), min=-0.5,max=0.5) #Twist of follower w.r.t ego vx
        self.actions[:,3] = torch.clamp(actions[:,3].clone(), min=-0.5,max=0.5) #Twist of follower w.r.t ego vf
        self.actions[:,4] = torch.clamp(actions[:,4].clone(), min=-1.0, max=1.0) #Twist of follower w.r.t ego theta_dot
        
        # V_follower = ((Vex + Vey) - (B+D))/(A-C) # Adjoint transformation for converting relative twist to body twist
        theta_e = self.diablo.data.heading_w
        theta_f = self.diablo2.data.heading_w

        xe = self.diablo.data.root_pos_w[:,0]
        ye = self.diablo.data.root_pos_w[:,1]

        xf = self.diablo2.data.root_pos_w[:,0]
        yf = self.diablo2.data.root_pos_w[:,1]
        theta_dot = self.actions[:,4]

        vx = self.actions[:,2]
        vy = self.actions[:,3]

        A = torch.cos(theta_e - theta_f)
        B = theta_dot*(yf*torch.cos(theta_e) + xe*torch.sin(theta_e) 
                       - ye*torch.cos(theta_e) - xf*torch.sin(theta_e) 
        )
        C = torch.sin(theta_f - theta_e)
        D = theta_dot*(xe*torch.cos(theta_e) - xf*torch.cos(theta_e)
                       + ye*torch.sin(theta_e) - yf*torch.sin(theta_e)

        )
        self.v_follower = torch.clamp(((vx + vy) - (B+D))/(A+C), min=-0.5, max=0.5)
        

    def _apply_action(self) -> None:
        omega_r_diabloego = (2*self.actions[:,0] + self.actions[:,1]*0.4825)/(2*0.0934)  # 0.4825 - Track Width, 0.0934 - Wheel radius
        omega_l_diabloego = (2*self.actions[:,0] - self.actions[:,1]*0.4825)/(2*0.0934)

        omega_r_diablofollower = (2*self.v_follower + self.actions[:,4]*0.4825)/(2*0.0934)
        omega_l_diablofollower = (2*self.v_follower - self.actions[:,4]*0.4825)/(2*0.0934)

        self.diablo.set_joint_velocity_target(omega_l_diabloego.unsqueeze(-1), joint_ids=self._leftwheel_dof_idx)
        self.diablo.set_joint_velocity_target(omega_r_diabloego.unsqueeze(-1), joint_ids=self._rightwheel_dof_idx)
                
        self.diablo2.set_joint_velocity_target(omega_l_diablofollower.unsqueeze(-1), joint_ids=self._leftwheel_dof_idx_f)
        self.diablo2.set_joint_velocity_target(omega_r_diablofollower.unsqueeze(-1), joint_ids=self._rightwheel_dof_idx_f)
        
    def _get_observations(self) -> dict:
        self._previous_actions = self.actions.clone()
        heading_angle_box = self.object.data.heading_w
        self.distances_robots = torch.sqrt(torch.sum((self.diablo.data.root_pos_w[:,0:2] - self.diablo2.data.root_pos_w[:,0:2])**2, dim=-1)).unsqueeze(-1)
        self.translated_coordinates = self.translated_coordinates.to(self.device)
        coordinates_box = self.object.data.root_pos_w[:,0:2].unsqueeze(1)    
        distances = torch.sqrt(torch.sum((coordinates_box - self.translated_coordinates) ** 2, dim=-1))
        min_distance_idx = torch.argmin(distances, dim=1)
        min_distance_idx_1 = torch.clamp(min_distance_idx+1, min=0, max=self.translated_coordinates.size(1) - 1)
        desired_heading = self.translated_coordinates[torch.arange(distances.shape[0]), min_distance_idx_1] - \
                          self.translated_coordinates[torch.arange(distances.shape[0]), min_distance_idx]
        desired_heading_angle = torch.atan2(desired_heading[:,1], desired_heading[:,0])
        heading_angle_Error = desired_heading_angle - heading_angle_box

        self.crosstrack_error = distances[torch.arange(distances.shape[0]), min_distance_idx]
        x_d1 = self.diablo.data.root_pos_w[:,0]
        y_d1 = self.diablo.data.root_pos_w[:,1]
        theta_d1 = self.diablo.data.heading_w
        
        x_o = self.object.data.root_pos_w[:,0]
        y_o = self.object.data.root_pos_w[:,1]
        theta_o = self.object.data.heading_w

        x_d2 = self.diablo2.data.root_pos_w[:,0]
        y_d2 = self.diablo2.data.root_pos_w[:,1]
        theta_d2 = self.diablo2.data.heading_w
      
        relative_pos_dox = x_o*torch.cos(theta_d1)  - x_d1*torch.cos(theta_d1) - y_d1*torch.sin(theta_d1) + y_o*torch.sin(theta_d1) # Relative pose (x-y) of ego w.r.t the payload
        relative_pos_doy = y_o*torch.cos(theta_d1)  - y_d1*torch.cos(theta_d1) + x_d1*torch.sin(theta_d1) - x_o*torch.sin(theta_d1)

        relative_pos_d2ox = x_d2*torch.cos(theta_o)  - x_o*torch.cos(theta_o) - y_o*torch.sin(theta_o) + y_d2*torch.sin(theta_o) # Relative pose (x-y) of follower w.r.t the payload
        relative_pos_d2oy = y_d2*torch.cos(theta_o)  - y_o*torch.cos(theta_o) + x_o*torch.sin(theta_o) - x_d2*torch.sin(theta_o)

        relative_heading_d1 = theta_d1 - theta_o # Relative angle of ego w.r.t the payload
        relative_heading_d2 = theta_o - theta_d2 # Relative angle of follower w.r.t the payload
 
        relative_heading_d1d2 = theta_d1 - theta_d2 # Relative pose of follower w.r.t ego
        relative_pos_d1d2x = x_d2*torch.cos(theta_d1)  - x_d1*torch.cos(theta_d1) - y_d1*torch.sin(theta_d1) + y_d2*torch.sin(theta_d1)
        relative_pos_d1d2y = y_d2*torch.cos(theta_d1)  - y_d1*torch.cos(theta_d1) + x_d1*torch.sin(theta_d1) - x_d2*torch.sin(theta_d1)


        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.distances_robots,
                    heading_angle_Error.unsqueeze(-1),
                    self.crosstrack_error.unsqueeze(-1),
                    self._previous_actions,
                    relative_heading_d1.unsqueeze(-1),
                    relative_heading_d2.unsqueeze(-1),
                    relative_pos_dox.unsqueeze(-1),
                    relative_pos_doy.unsqueeze(-1),
                    relative_pos_d2ox.unsqueeze(-1),
                    relative_pos_d2oy.unsqueeze(-1),
                    relative_heading_d1d2.unsqueeze(-1),
                    relative_pos_d1d2x.unsqueeze(-1),
                    relative_pos_d1d2y.unsqueeze(-1)
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations    
        
    def _get_rewards(self) -> torch.Tensor:
        coordinates_box = self.object.data.root_pos_w[:,0:2].unsqueeze(1)    
        self.distances = torch.sqrt(torch.sum((coordinates_box - self.translated_coordinates) ** 2, dim=-1))
        min_distance_idx = torch.argmin(self.distances, dim=1)
        min_distance_idx_1 = torch.clamp(min_distance_idx+1, min=0, max=self.translated_coordinates.size(1) - 1)
        desired_heading = self.translated_coordinates[torch.arange(self.distances.shape[0]), min_distance_idx_1] - \
                          self.translated_coordinates[torch.arange(self.distances.shape[0]), min_distance_idx]
        desired_heading_angle = torch.atan2(desired_heading[:,1], desired_heading[:,0])

        box_speed = self.object.data.root_lin_vel_b[:,0]  

        heading_angle_box = self.object.data.heading_w
         
        heading_angle_Error = torch.abs(desired_heading_angle - heading_angle_box)

        crosstrack_error = torch.abs(self.distances[torch.arange(self.distances.shape[0]), min_distance_idx])

        heading_angle_Error -= 0.0
        heading_angle_Error /= math.pi

        diabo_1 = self.diablo.data.root_pos_w[:,0:2] # Ego
        
        diablo_2 = self.diablo2.data.root_pos_w[:,0:2] # Follower
        
        dist_diablo = torch.sqrt(torch.sum((diabo_1 - diablo_2) ** 2, dim=-1))

        action_diff = torch.norm((self.actions - self._previous_actions), dim=1)
       
        reward_large_actions = torch.exp(-0.05*action_diff)
        reward_box_crosstrack = torch.exp(-7.0*crosstrack_error)
        reward_headingangle = torch.exp(-3.0*heading_angle_Error)
        reward_dist_diablo = torch.where((dist_diablo <= 0.5) | (dist_diablo >= 1.75), -1.0, 0.0)
        reward_speed = box_speed
    
        total_reward = 4.0*reward_box_crosstrack*reward_headingangle*reward_speed*reward_large_actions + \
                        4.0*reward_dist_diablo -10.0*self.box_reset_criteria -10.0*self.crosstrack_error_criteria
       
        
        total_reward = torch.clip(total_reward, 0.0, None)
        
        return total_reward    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        coordinates_box = self.object.data.root_pos_w[:,0:2].unsqueeze(1) 
        self.distances = torch.sqrt(torch.sum((coordinates_box - self.translated_coordinates) ** 2, dim=-1))
        
        self.coordinate_box_z = self.object.data.root_pos_w[:,2]
        diabo_1 = self.diablo.data.root_pos_w[:,0:2]
        diablo_2 = self.diablo2.data.root_pos_w[:,0:2]
        dist_diablo = torch.sqrt(torch.sum((diabo_1 - diablo_2) ** 2, dim=-1))
        end_point_distance = self.distances[torch.arange(self.distances.shape[0]), self.translated_coordinates.size(1) - 1]

        self.end_point_criteria = torch.where(end_point_distance <= 0.3, 1.0, 0.0).bool()

        robot_reset_criteria = torch.where((dist_diablo >= 1.75) | (dist_diablo <= 0.5), 1.0, 0.0).bool()
        self.box_reset_criteria = torch.where(self.coordinate_box_z <= 0.2, 1.0, 0.0).bool()
        self.crosstrack_error_criteria = torch.where(torch.abs(self.crosstrack_error) >= 0.5, 1.0,0.0).bool()
        self.terminate = self.box_reset_criteria | robot_reset_criteria | self.crosstrack_error_criteria | self.end_point_criteria
        
        return self.terminate, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        env_ids2 = env_ids
        env_ids3 = env_ids
        if env_ids is None:
            env_ids = self.diablo._ALL_INDICES
            env_ids2 = self.diablo2._ALL_INDICES
            env_ids3 = self.object._ALL_INDICES
        super()._reset_idx(env_ids)
        super()._reset_idx(env_ids2)
        super()._reset_idx(env_ids3)

        joint_pos = self.diablo.data.default_joint_pos[env_ids]
        joint_vel = self.diablo.data.default_joint_vel[env_ids]

        default_root_state = self.diablo.data.default_root_state[env_ids]
        
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        joint_pos2 = self.diablo2.data.default_joint_pos[env_ids2]
                
        joint_vel2 = self.diablo2.data.default_joint_vel[env_ids2]

        default_root_state2 = self.diablo2.data.default_root_state[env_ids2]
        
        self.joint_pos2[env_ids2] = joint_pos2
        self.joint_vel2[env_ids2] = joint_vel2

        default_root_state3 = self.object.data.default_root_state[env_ids3]
        
        self.translated_coordinates = self.translated_coordinates.to(self.device)
        coordinates_box = self.object.data.root_pos_w[:,0:2].unsqueeze(1) 
        self.distances = torch.sqrt(torch.sum((coordinates_box - self.translated_coordinates) ** 2, dim=-1))
        self.distances_end = self.distances[env_ids3,self.translated_coordinates.size(1) - 1]
        
        default_root_state[:,0] = random.uniform(-0.2,0.2)
        default_root_state[:,1] = random.uniform(-0.2,0.2)
        
        default_root_state2[:,0] = default_root_state[:,0] - 0.55
        default_root_state2[:,1] = default_root_state[:,1]
        

        default_root_state3[:, 0] = (default_root_state[:,0] + default_root_state2[:,0])/2
        default_root_state3[:, 1] = default_root_state[:,1] + random.uniform(-0.1,0.1)
        default_root_state3[:, 2] = random.uniform(0.57,0.75)
                              
        
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state2[:, :3] += self.scene.env_origins[env_ids2]
        default_root_state3[:, :3] += self.scene.env_origins[env_ids3]

        self.diablo.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.diablo.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.diablo.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.diablo2.write_root_pose_to_sim(default_root_state2[:, :7], env_ids2)
        self.diablo2.write_root_velocity_to_sim(default_root_state2[:, 7:], env_ids2)
        self.diablo2.write_joint_state_to_sim(joint_pos2, joint_vel2, None, env_ids2)
        
        self.object.write_root_pose_to_sim(default_root_state3[:, :7], env_ids3)
        self.object.write_root_velocity_to_sim(default_root_state3[:, 7:], env_ids3)
        