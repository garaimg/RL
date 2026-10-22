# PushBox

`PushBoxEnv` is a deliberately small Gymnasium environment for tabular
reinforcement learning. The MDP has discrete states and actions. The ANSI,
PyGame, and MuJoCo renderers are separate views of that state and do not
participate in the transition logic.

The agent and goal have fixed corner positions. The grid boundary behaves like
an invisible wall and does not occupy or display any cells. The box shown below
is one possible starting position:

```text
A.....
......
..B...
......
......
.....G
```

On every reset, the box is sampled one cell away from the grid boundary.
The sampling uses Gymnasium's seeded random-number generator, so repeated calls
to `reset(seed=the_same_seed)` produce the same initial box position.

Actions are `0 = UP`, `1 = RIGHT`, `2 = DOWN`, and `3 = LEFT`. Every
non-terminal transition gives `-0.01`, including attempts blocked by the grid
boundary. Putting the box on `G` terminates the episode and gives `+1.0`.

## State encoding

Each `(x, y)` position is first converted to a row-major cell index:

```python
cell_index = y * width + x
```

The two cell indices are then combined:

```python
state = agent_index * number_of_cells + box_index
```

The observation space includes states where the agent and box overlap, although
the transition dynamics never produce them. This modest use of extra Q-table
rows keeps encoding and decoding straightforward enough to derive on the board.

## Running the material

From the repository root, install the dependencies and run:

```bash
python -m pip install -r push_box/requirements.txt
python -m push_box.examples.random_agent
pytest push_box/tests
```

For the PyGame renderer, install its optional dependency and run:

```bash
python -m pip install -r push_box/requirements-pygame.txt
python -m push_box.examples.random_agent --human
```

Both examples use ANSI by default. Pass `--human` to use PyGame:

```bash
python -m push_box.examples.random_agent --human
python -m push_box.examples.train_qlearning --human
```

For the MuJoCo renderer:

```bash
python -m pip install -r push_box/requirements-mujoco.txt
python -m push_box.examples.random_agent --mujoco
python -m push_box.examples.train_qlearning --mujoco
```

`PushBoxPhysicsEnv` inherits directly from Gymnasium's `MujocoEnv` and
contains a `PushBoxEnv`. It keeps the same discrete MDP, but uses forces,
contacts, mass, and friction while moving between cell centres. At the end of
each action it places the bodies exactly at their target cells and sets their
velocities to zero:

```bash
python -m push_box.examples.random_agent_physics_env
python -m push_box.examples.train_qlearning_physics_env
```

The environment uses `render_mode="ansi"` for text and `render_mode="human"`
for graphical output. The graphical backend is selected with
`renderer="pygame"` or `renderer="mujoco"`. Rendering is explicit: use
`print(env.render())` for ANSI or `env.render()` for a graphical backend.

The Q-learning example also needs the dependencies already used by the course's
`deustorl` package:

```bash
python -m pip install -r push_box/requirements-qlearning.txt
python -m push_box.examples.train_qlearning
```

It imports the existing `deustorl/qlearning.py` and uses `evaluate_policy` from
`deustorl/common.py` for rendered evaluation.

The ANSI, PyGame, and MuJoCo renderers only consume the positions produced by
`PushBoxEnv`; they never calculate the next state. The separate
`PushBoxPhysicsEnv` does run MuJoCo physics between states, while
`PushBoxEnv.step()` still determines the final discrete transition.
