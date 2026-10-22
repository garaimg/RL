"""PushBox with discrete decisions and physical transitions."""

from typing import Any

import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv

from push_box.pushbox_env import Position, PushBoxEnv


class PushBoxPhysicsEnv(MujocoEnv):
    """Use forces to animate each transition between discrete grid states."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
        "render_fps": 200,
    }

    POSITION_GAIN = 40.0
    VELOCITY_GAIN = 8.0
    MAX_FORCE = 20.0
    MAX_PHYSICS_STEPS = 200
    RENDER_EVERY_N_STEPS = 6

    MODEL_XML = """
<mujoco model="pushbox_physics">
    <option timestep="0.005" gravity="0 0 -9.81"/>
    <asset>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1="0.9 0.9 0.85" rgb2="0.75 0.75 0.7"
                 width="512" height="512"/>
        <material name="grid" texture="grid"
                  texrepeat="6 6" texuniform="true"/>
    </asset>
    <default>
        <geom friction="1.0 0.005 0.0001" solref="0.01 1"/>
    </default>
    <worldbody>
        <light pos="0 0 10"/>
        <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>

        <!-- Invisible boundary walls. -->
        <geom type="box" pos="-3.1 0 0.5" size="0.1 3 0.5" rgba="0 0 0 0"/>
        <geom type="box" pos="3.1 0 0.5" size="0.1 3 0.5" rgba="0 0 0 0"/>
        <geom type="box" pos="0 -3.1 0.5" size="3 0.1 0.5" rgba="0 0 0 0"/>
        <geom type="box" pos="0 3.1 0.5" size="3 0.1 0.5" rgba="0 0 0 0"/>

        <body name="goal" pos="0 0 0.025">
            <geom type="cylinder" size="0.4 0.025" rgba="0.2 0.8 0.3 1"
                  contype="0" conaffinity="0"/>
        </body>
        <body name="box" pos="0 0 0.48">
            <joint name="box_x" type="slide" axis="1 0 0" damping="4.0"/>
            <joint name="box_y" type="slide" axis="0 1 0" damping="4.0"/>
            <geom type="box" size="0.48 0.48 0.48" mass="1.5"
                  rgba="0.75 0.4 0.1 1"/>
        </body>
        <body name="agent" pos="0 0 0.48">
            <joint name="agent_x" type="slide" axis="1 0 0" damping="2.0"/>
            <joint name="agent_y" type="slide" axis="0 1 0" damping="2.0"/>
            <geom type="cylinder" size="0.48 0.48" mass="1.0"
                  rgba="0.1 0.4 0.9 1"/>
        </body>
    </worldbody>
    <actuator>
        <motor name="agent_x_force" joint="agent_x" ctrlrange="-20 20"/>
        <motor name="agent_y_force" joint="agent_y" ctrlrange="-20 20"/>
    </actuator>
</mujoco>
"""

    def __init__(self, render_mode: str | None = None, max_steps: int = 200):
        self.game = PushBoxEnv(max_steps=max_steps)
        self._reset_seed = None
        self._reset_options = None
        self._reset_info = {}

        camera = {
            "lookat": np.array([0.0, 0.0, 0.0]),
            "distance": 8.4,
            "azimuth": 135.0,
            "elevation": -55.0,
        }

        super().__init__(
            model_path=__file__,
            frame_skip=1,
            observation_space=self.game.observation_space,
            render_mode=render_mode,
            width=640,
            height=640,
            default_camera_config=camera,
        )

        self.action_space = self.game.action_space
        self._copy_game_state()

    def _initialize_simulation(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        model = mujoco.MjModel.from_xml_string(self.MODEL_XML)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
        return model, mujoco.MjData(model)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        self._reset_seed = seed
        self._reset_options = options
        return super().reset(seed=seed, options=options)

    def reset_model(self) -> int:
        observation, self._reset_info = self.game.reset(
            seed=self._reset_seed,
            options=self._reset_options,
        )
        self._copy_game_state()
        return observation

    def _get_reset_info(self) -> dict[str, Any]:
        return self._reset_info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        previous_agent_position = self.game.agent_position
        previous_box_position = self.game.box_position
        result = self.game.step(action)

        agent_moved = self.game.agent_position != previous_agent_position
        box_moved = self.game.box_position != previous_box_position

        if agent_moved:
            self._simulate_transition(box_moved)

        self._copy_game_state()

        if self.render_mode == "human":
            self.render()

        return result

    def _simulate_transition(self, box_moved: bool) -> None:
        target_agent = self._world_position(self.game.agent_position)
        target_box = self._world_position(self.game.box_position)

        for physics_step in range(self.MAX_PHYSICS_STEPS):
            agent_position = self._body_position("agent")
            agent_velocity = self._body_velocity("agent")
            position_error = target_agent - agent_position

            force = self.POSITION_GAIN * position_error - self.VELOCITY_GAIN * agent_velocity
            force = np.clip(force, -self.MAX_FORCE, self.MAX_FORCE)
            self.do_simulation(force, n_frames=1)

            if self.render_mode == "human" and physics_step % self.RENDER_EVERY_N_STEPS == 0:
                self.render()

            agent_arrived = np.linalg.norm(target_agent - self._body_position("agent")) < 0.03
            box_arrived = not box_moved or np.linalg.norm(target_box - self._body_position("box")) < 0.06
            if agent_arrived and box_arrived:
                break

    def _copy_game_state(self) -> None:
        self._set_body_position("agent", self.game.agent_position)
        self._set_body_position("box", self.game.box_position)
        self.model.body("goal").pos[:2] = self._world_position(self.game.GOAL_POSITION)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_body_position(self, body_name: str, position: Position) -> None:
        world_position = self._world_position(position)
        for axis, coordinate in zip(("x", "y"), world_position):
            qpos_address = int(self.model.joint(f"{body_name}_{axis}").qposadr[0])
            self.data.qpos[qpos_address] = coordinate

    def _body_position(self, body_name: str) -> np.ndarray:
        return np.array([
            self.data.qpos[int(self.model.joint(f"{body_name}_x").qposadr[0])],
            self.data.qpos[int(self.model.joint(f"{body_name}_y").qposadr[0])],
        ])

    def _body_velocity(self, body_name: str) -> np.ndarray:
        return np.array([
            self.data.qvel[int(self.model.joint(f"{body_name}_x").dofadr[0])],
            self.data.qvel[int(self.model.joint(f"{body_name}_y").dofadr[0])],
        ])

    def _world_position(self, position: Position) -> np.ndarray:
        x, y = position
        return np.array([
            x - (self.game.WIDTH - 1) / 2,
            (self.game.HEIGHT - 1) / 2 - y,
        ])

    def close(self) -> None:
        super().close()
        self.game.close()
