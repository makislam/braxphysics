import jax
from jax import numpy as jp
import brax
from brax.io import html
from brax.io import mjcf

# In v2, we select our physics backend explicitly. 
# 'generalized' is the most accurate for robotics (similar to your old script).
from brax.generalized import pipeline

# 1. Define the world using MJCF (standard MuJoCo XML)
# This replaces the old "bodies { ... }" config.
mjcf_string = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="20 20 0.5"/>

    <body name="ball" pos="0 0 5">
      <freejoint/>
      <geom type="sphere" size="0.5" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Load the system
sys = mjcf.loads(mjcf_string)

# 2. Initialize the State
# In v2, pipeline.init automatically handles the initial positions defined in the XML.
# Since we set pos="0 0 5" in the XML, we don't need to manually set it here!
initial_velocity = jp.array([5, 0, 0, 0, 10, 0])  # Initial velocity: [vx, vy, vz, wx, wy, wz]
state = jax.jit(pipeline.init)(sys, sys.init_q, initial_velocity)

# 3. Run the Simulation
# We define a step function using the generalized pipeline
step_fn = jax.jit(pipeline.step)

states = []
for i in range(5000):
    # pipeline.step takes (sys, state, actions). We pass 0 actions (no motors).
    state = step_fn(sys, state, jp.zeros(sys.act_size()))
    states.append(state)

# 4. Export to HTML
# The visualizer works mostly the same way
with open("ball_sim_v2.html", "w") as f:
    f.write(html.render(sys, states))

print("Done! Created 'ball_sim_v2.html' using Brax v2.")