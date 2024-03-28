import numpy as np
from numpy.linalg import norm
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import torch

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class HumanoidEnv_v3(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._graph_dict = {"root":["abdomen_x", "abdomen_y", "abdomen_z",
                                    "right_shoulder1", "right_shoulder2",
                                    "left_shoulder1", "left_shoulder2"], 
                            "abdomen_x":["right_hip_x", "right_hip_y", "right_hip_z", 
                            "left_hip_x", "left_hip_y", "left_hip_z" ],
                            "abdomen_y":["right_hip_x", "right_hip_y", "right_hip_z", 
                            "left_hip_x", "left_hip_y", "left_hip_z" ],
                            "abdomen_z":["right_hip_x", "right_hip_y", "right_hip_z", 
                            "left_hip_x", "left_hip_y", "left_hip_z" ],
                            "right_hip_x":["right_knee"],
                            "right_hip_y":["right_knee"],
                            "right_hip_z":["right_knee"],
                            "left_hip_x":["left_knee"],
                            "left_hip_y":["left_knee"],
                            "left_hip_z":["left_knee"],
                            "right_knee":[],
                            "left_knee":[],
                            "right_shoulder1":["right_elbow"],
                            "right_shoulder2":["right_elbow"],
                            "left_shoulder1":["left_elbow"],
                            "left_shoulder2":["left_elbow"],
                            "right_elbow":[],
                            "left_elbow":[]}
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,2,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,2,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        all_node_feats = []
        for node in self._graph_dict:  
            
            node_feats = []
            # Obs 1: joint's pos
            node_feats.append(self.data.joint(node).qpos[0])
           
            # Obs 2: joint's vel
            node_feats.append(self.data.joint(node).qvel[0])
            
            all_node_feats.append(node_feats)
        
        # all_node_feats = np.asarray(all_node_feats)
        
        return all_node_feats

    def get_edge_index(self):
        graph_dict = self._graph_dict
        # returns the edge_index: list of tuples with connected joints in a graph
        edge_index = []

        for node in graph_dict:
            node_idx = list(graph_dict).index(node)
        
            for neighbor in graph_dict[node]:
                neighbor_idx = list(graph_dict).index(neighbor)
            
                edge_index += [[node_idx, neighbor_idx]]
                
        edge_index = torch.tensor(edge_index)
        
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        
        return edge_index
    
    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        xy_vector_before = self.data.qpos
        
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)
        xy_vector_after = self.data.qpos
        
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        
        # calculate similarity reward between poses using Similarity = (A.B) / (||A||.||B||) 
        similarity_reward = np.dot(xy_vector_before,xy_vector_after)/(norm(xy_vector_before)*norm(xy_vector_after))
        
        rewards = forward_reward + healthy_reward + similarity_reward

        observation = self._get_obs()
        
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "similarity_reward": similarity_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        
        return observation