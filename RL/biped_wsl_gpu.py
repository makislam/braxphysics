from jax import numpy as jp
from brax import envs
from brax.io import html
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

env_name = 'humanoid'
env = envs.get_environment(env_name=env_name, backend='generalized')

import jax
print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("Default backend:", jax.default_backend())

# PPO (Proximal Policy Optimization)
def make_networks_factory(obs_shape, action_size, preprocess_observations_fn=lambda x: x):
    return ppo_networks.make_ppo_networks(
        obs_shape, action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(128, 128, 128, 128),
        value_hidden_layer_sizes=(128, 128, 128, 128),
    )

# Train

print("Training started...")
train_fn = lambda: ppo.train(
    environment=env,
    num_timesteps=50_000_000, #total simulation steps
    num_evals=10,
    reward_scaling=0.1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=2048,
    batch_size=1024,
    network_factory=make_networks_factory,
)

# Run the training on the GPU
inference_fn, params, metrics = train_fn()
print("Training Complete!")

# Visualize

# Initialize a fresh environment for playback
env = envs.create(env_name=env_name, backend='generalized')
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference = jax.jit(inference_fn)

# Start the robot
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
states = []

# Run for 500 steps
for _ in range(500):
    rng, rng_act = jax.random.split(rng)
    # Ask the AI what to do based on the current state
    act_rng, _ = jax.random.split(rng_act)
    act, _ = jit_inference(params, state.obs, act_rng)
    
    # Step the physics
    state = jit_step(state, act)
    states.append(state.pipeline_state)

# Save to HTML
with open("humanoid_walk.html", "w") as f:
    f.write(html.render(env.sys, states))

print("Check 'humanoid_walk.html' to see your robot walk!")