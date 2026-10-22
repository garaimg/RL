"""MuJoCo visualization for the discrete PushBox state."""

import time

import mujoco
import mujoco.viewer

Position = tuple[int, int]


class MujocoRenderer:
    """Copy discrete grid positions to simple 3D objects."""

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.model = mujoco.MjModel.from_xml_string(self._create_model_xml())
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def render(self, *, agent_position: Position, box_position: Position, goal_position: Position) -> None:
        if self.viewer is None:
            self._update_positions(agent_position, box_position, goal_position)
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = (0, 0, 0)
                self.viewer.cam.distance = max(self.width, self.height) * 1.4
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -55
        elif self.viewer.is_running():
            with self.viewer.lock():
                self._update_positions(agent_position, box_position, goal_position)

        if self.viewer.is_running():
            self.viewer.sync()
            time.sleep(1 / self.fps)

    def _update_positions(self, agent_position: Position, box_position: Position, goal_position: Position) -> None:
        self._set_position("agent", agent_position, height=0.35)
        self._set_position("box", box_position, height=0.32)
        self._set_position("goal", goal_position, height=0.03)
        mujoco.mj_forward(self.model, self.data)

    def _set_position(self, body_name: str, position: Position, height: float) -> None:
        x, y = position
        world_x = x - (self.width - 1) / 2
        world_y = (self.height - 1) / 2 - y
        mocap_id = int(self.model.body(body_name).mocapid[0])
        self.data.mocap_pos[mocap_id] = (world_x, world_y, height)

    def _create_model_xml(self) -> str:
        return f"""
<mujoco model="pushbox">
    <asset>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1="0.9 0.9 0.85" rgb2="0.75 0.75 0.7"
                 width="512" height="512"/>
        <material name="grid" texture="grid"
                  texrepeat="{self.width} {self.height}" texuniform="true"/>
    </asset>
    <worldbody>
        <light pos="0 0 10"/>
        <geom type="plane" size="{self.width / 2} {self.height / 2} 0.1"
              material="grid" contype="0" conaffinity="0"/>

        <body name="goal" mocap="true">
            <geom type="cylinder" size="0.4 0.025" rgba="0.2 0.8 0.3 1"/>
        </body>
        <body name="box" mocap="true">
            <geom type="box" size="0.32 0.32 0.32" rgba="0.75 0.4 0.1 1"/>
        </body>
        <body name="agent" mocap="true">
            <geom type="sphere" size="0.35" rgba="0.1 0.4 0.9 1"/>
        </body>
    </worldbody>
</mujoco>
"""

    def close(self) -> None:
        if self.viewer is not None:
            viewer = self.viewer
            self.viewer = None
            viewer.close()

            # close() only sends a stop request to MuJoCo's viewer thread.
            # Wait briefly for that thread to release the OpenGL window before
            # Python exits and GLFW destroys the display connection.
            deadline = time.monotonic() + 2.0
            while viewer.m is not None and time.monotonic() < deadline:
                time.sleep(0.01)
