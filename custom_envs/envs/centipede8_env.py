import numpy as np
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

class Centipede8Env(MujocoEnv, utils.EzPickle):
    """
        @brief:
            In the CentipedeEnv, we define a task that is designed to test the
            power of transfer learning for gated graph neural network.
        @children:
            @CentipedeFourEnv
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, CentipedeLegNum=8, **kwargs):

        xml_path = "/home/ai/Desktop/2023/fall23/thesis/ppo/custom_envs/envs/CentipedeEight.xml"
        self.num_body = int(np.ceil(CentipedeLegNum / 2.0))
        self._control_cost_coeff = 0.5 * 4 / CentipedeLegNum
        self._contact_cost_coeff = 0.5 * 1e-3 * 4 / CentipedeLegNum

        self._n_legs = CentipedeLegNum

        self.torso_geom_id = 1 + np.array(list(range(self.num_body))) * 5
        # make sure the centipede is not born to be end of episode
        self.body_qpos_id = 6 + 6 + np.array(list(range(self.num_body))) * 6
        self.body_qpos_id[-1] = 5
        
        observation_space = Box(
                low=-np.inf, high=np.inf, shape=(97,), dtype=np.float64
            )
        
        MujocoEnv.__init__(
            self, 
            xml_path, 
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
            )
        utils.EzPickle.__init__(self, **kwargs)

    def get_env_num_str(self, number):
        num_str = num2words.num2words(number).replace("-", "")
        return num_str[0].upper() + num_str[1:]

    def step(self, a):
        xposbefore = self.get_body_com("torso_" + str(self.num_body - 1))[0]
        self.do_simulation(a, self.frame_skip)
        """
        xposafter = np.mean([self.get_body_com("torso_" + str(i_torso))[0]
                             for i_torso in range(self.num_body)])
        """
        xposafter = self.get_body_com("torso_" + str(self.num_body - 1))[0]

        # calculate reward
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = self._control_cost_coeff * np.square(a).sum()

        contact_cost = self._contact_cost_coeff * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # check if finished
        state = self.state_vector()
        notdone = (
            np.isfinite(state).all()
            and self._check_height()
            and self._check_direction()
        )
        done = not notdone

        ob = self._get_obs()

        return (
            ob,
            reward,
            done,
            False,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.data.qpos.flat[2:],
                self.data.qvel.flat,
                np.clip(self.data.cfrc_ext, -1, 1).flat,
            ])
        # print(obs.shape)
        
        return obs

    def reset_model(self):
        VEL_NOISE_MULT = 0.1
        POS_NOISE_MULT = 0.1
        while True:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-POS_NOISE_MULT, high=POS_NOISE_MULT
            )
            qpos[self.body_qpos_id] = self.np_random.uniform(
                size=len(self.body_qpos_id),
                low=-POS_NOISE_MULT / (self.num_body - 1),
                high=POS_NOISE_MULT / (self.num_body - 1),
            )

            qvel = self.init_qvel + self.np_random.random(self.model.nv) * VEL_NOISE_MULT
            self.set_state(qpos, qvel)
            if self._check_height() and self._check_direction():
                break

        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.mode = 'rgb_array'
        # self.viewer.cam.distance = self.model.stat.extent * 3
        # body_name = 'torso_' + str(int(np.ceil(self.num_body / 2 - 1)))
        # #self.viewer.cam.trackbodyid = self.model.body_names.index(body_name)
        # self.viewer.cam.trackbodyid = 4
        # #self.viewer.cam.lookat[2] = self.model.stat.center[2]
        # self.viewer.cam.elevation = -20
        # self.viewer.cam.azimuth = -20

        # self.viewer.cam.distance = self.model.stat.extent * 5.
        body_name = "torso_" + str(int(np.ceil(self.num_body / 2 - 1)))
        # self.viewer.cam.trackbodyid = self.model.body_names.index(body_name)
        # self.viewer.cam.trackbodyid = 10

    """
    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.5).all() and (height > 0.35).all()
    """

    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.15).all() and (height > 0.35).all()

    def _check_direction(self):
        y_pos_pre = self.data.geom_xpos[self.torso_geom_id[:-1], 1]
        y_pos_post = self.data.geom_xpos[self.torso_geom_id[1:], 1]
        y_diff = np.abs(y_pos_pre - y_pos_post)
        return (y_diff < 0.45).all()

