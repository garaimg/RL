"""Gymnasium environment containing the PushBox transition dynamics."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces

from push_box.renderers.ansi_renderer import render_ansi

Position = tuple[int, int]

class PushBoxEnv(gym.Env[int, int]):
    """A deterministic grid world in which an agent pushes one box to a goal.

    Coordinates are ``(x, y)`` pairs. ``x`` increases to the right and ``y``
    increases downward. Rendering reads the state but never changes it.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    ACTION_DIRECTIONS: dict[int, Position] = {
        UP: (0, -1),
        RIGHT: (1, 0),
        DOWN: (0, 1),
        LEFT: (-1, 0),
    }

    STEP_PENALTY = -0.01
    GOAL_REWARD = 1.0

    WIDTH = 6
    HEIGHT = 6
    START_AGENT_POSITION = (0, 0)
    GOAL_POSITION = (WIDTH - 1, HEIGHT - 1)

    def __init__(self, render_mode: str | None = None, renderer: str = "pygame", max_steps: int = 200):
        super().__init__()

        self.render_mode = render_mode
        self.renderer_name = renderer
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)

        # Each position, agent or box, has WIDTH * HEIGHT possible indices. States
        # where the agent and box overlap are unreachable, but keeping them
        # makes the encoding especially transparent.
        self.number_of_cells = self.WIDTH * self.HEIGHT
        self.observation_space = spaces.Discrete(self.number_of_cells**2)

        self.agent_position, self.box_position = self._initial_positions()
        self._elapsed_steps = 0
        self._renderer = None

    def encode_state(self, agent_pos: Position, box_pos: Position) -> int:
        """Encode ``(agent_x, agent_y, box_x, box_y)`` as one integer."""
        agent_index = agent_pos[1] * self.WIDTH + agent_pos[0]
        box_index = box_pos[1] * self.WIDTH + box_pos[0]
        return agent_index * self.number_of_cells + box_index

    def decode_state(self, state: int) -> tuple[Position, Position]:
        """Decode an integer state into ``(agent_position, box_position)``."""
        if not self.observation_space.contains(state):
            raise ValueError(f"state must be in {self.observation_space}")

        agent_index, box_index = divmod(int(state), self.number_of_cells)
        agent_y, agent_x = divmod(agent_index, self.WIDTH)
        box_y, box_x = divmod(box_index, self.WIDTH)
        return (agent_x, agent_y), (box_x, box_y)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        """Reset the agent and goal, and sample a safe box position.

        The box uses Gymnasium's seeded random-number generator, so calling
        ``reset(seed=...)`` makes its position reproducible.
        """
        super().reset(seed=seed)

        self.agent_position, self.box_position = self._initial_positions()
        self._elapsed_steps = 0

        observation = self.encode_state(self.agent_position, self.box_position)
        return observation, {}

    def _initial_positions(self) -> tuple[Position, Position]:
        box_x = int(self.np_random.integers(1, self.WIDTH - 1))
        box_y = int(self.np_random.integers(1, self.HEIGHT - 1))
        box_position = (box_x, box_y)
        return self.START_AGENT_POSITION, box_position

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Apply one of the four discrete movement actions."""

        direction_x, direction_y = self.ACTION_DIRECTIONS[int(action)]
        new_agent_position = (
            self.agent_position[0] + direction_x,
            self.agent_position[1] + direction_y,
        )

        if new_agent_position == self.box_position:
            new_box_position = (
                self.box_position[0] + direction_x,
                self.box_position[1] + direction_y,
            )

            if self._is_free(new_box_position):
                self.box_position = new_box_position
                self.agent_position = new_agent_position
        elif self._is_free(new_agent_position):
            self.agent_position = new_agent_position

        self._elapsed_steps += 1

        terminated = self.box_position == self.GOAL_POSITION
        truncated = self._elapsed_steps >= self.max_steps and not terminated
        reward = self.calculate_reward()

        observation = self.encode_state(self.agent_position, self.box_position)
        return observation, reward, terminated, truncated, {}

    def calculate_reward(self) -> float:
        """Calculate the reward associated with the current state."""
        if self.box_position == self.GOAL_POSITION:
            return self.GOAL_REWARD
        return self.STEP_PENALTY

    def _is_free(self, position: Position) -> bool:
        """Return whether a cell can receive the agent or box."""
        x, y = position
        inside_grid = 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT
        return inside_grid and position != self.box_position

    def render(self) -> str | None:
        """Render the current state without participating in its dynamics."""
        if self.render_mode == "ansi":
            return render_ansi(
                width=self.WIDTH,
                height=self.HEIGHT,
                agent_position=self.agent_position,
                box_position=self.box_position,
                goal_position=self.GOAL_POSITION,
            )

        if self.render_mode == "human":
            if self._renderer is None:
                if self.renderer_name == "pygame":
                    from push_box.renderers.pygame_renderer import PygameRenderer

                    self._renderer = PygameRenderer(
                        self.WIDTH, self.HEIGHT, self.metadata["render_fps"]
                    )
                elif self.renderer_name == "mujoco":
                    from push_box.renderers.mujoco_renderer import MujocoRenderer

                    self._renderer = MujocoRenderer(
                        self.WIDTH, self.HEIGHT, self.metadata["render_fps"]
                    )
                else:
                    raise ValueError(f"Unknown renderer: {self.renderer_name}")

            self._renderer.render(
                agent_position=self.agent_position,
                box_position=self.box_position,
                goal_position=self.GOAL_POSITION,
            )

        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
