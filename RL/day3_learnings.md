# Day 3 Learnings: Humanoid Walking with MoCap Imitation

**Date:** January 10, 2026  
**Best Version:** v20 (Anti-Tilt / Gyroscope Fix)

---

## 🎯 Goal
Train a humanoid robot to walk naturally using MoCap imitation learning with Brax/JAX.

---

## 📊 Version History

| Version | Key Changes | Result |
|---------|-------------|--------|
| v19 | Natural walking physics (CoM tracking, soft contact) | Good but torso tilting |
| **v20** | **Gyroscope reward + Death by tilt** | **✅ BEST - stable upright walking** |
| v21 | Extended training (100M steps, 1500 ep length) | ❌ Locked knee - survival strategy |
| v22 | Knee bend reward during swing phase | Testing with v20 params |

---

## ✅ What Worked Best (v20)

### Training Parameters
```python
num_timesteps = 30_000_000      # 30M steps (~15-25 min)
episode_length = 500            # Standard length
num_evals = 20
num_envs = 512
batch_size = 256
learning_rate = 3e-4
```

### Key Reward Components

1. **GYROSCOPE UPRIGHT REWARD** (The Key Fix)
   ```python
   upright_reward_weight = 5.0
   
   # Uses brax.math.rotate to compute world-space up vector
   def _compute_torso_uprightness(self, pipeline_state):
       torso_quat = pipeline_state.x.rot[self._torso_idx]
       local_up = jp.array([0.0, 0.0, 1.0])
       world_up = brax_math.rotate(local_up, torso_quat)
       return world_up[2]  # 1.0 = perfectly upright
   
   # Sharp exponential penalty for any tilt
   upright_reward = upright_weight * jp.exp(-20.0 * tilt)
   ```

2. **DEATH BY TILT TERMINATION**
   ```python
   max_tilt_angle = 0.8  # cos(35°) ≈ 0.82
   
   # Terminate if torso tilts > ~35 degrees
   is_tilted = jp.where(torso_uprightness < max_tilt_angle, 1.0, 0.0)
   ```

3. **Other Important Weights**
   ```python
   forward_reward_weight = 5.0    # Encourage forward motion
   posture_weight = 15.0          # Keep upright posture
   healthy_reward = 5.0           # Survival bonus
   rom_reward_weight = 5.0        # Range of motion (natural gait)
   side_lean_penalty = 5.0        # Reduced from 25.0
   ```

---

## ❌ What Didn't Work

### Extended Training (v21 - 100M steps, 1500 episode length)
**Problem:** Agent learned to "game" the reward function
- Froze knee to avoid side lean penalty over long episodes
- Prioritized survival over natural motion
- "Locked knee" survival strategy

**Diagnosis:**
- Long episodes (1500 steps) created too much survival pressure
- Agent found that Postural Stability > Imitation Accuracy
- Reference MoCap data may not loop perfectly for 1500 steps

### Lesson Learned
> Training straight for 1500 steps from scratch is brutal. The gradients for "good walking" at step 100 get washed out by the survival noise at step 1400.

---

## 🔧 Technical Details

### Environment: MoCapHumanoidLegsOnly
- Fixed arms at sides (only train legs)
- 11 action dimensions (leg joints only)
- CMU MoCap reference data (6 subjects averaged)

### MoCap Processing
```python
# Zero abdomen joints (keep torso upright)
swapped_ref[:, 0:3] = 0.0  # abdomen_y, abdomen_z, abdomen_x

# Zero hip rotation (legs point forward)
swapped_ref[:, 3] = 0.0    # right_hip_x
swapped_ref[:, 4] = 0.0    # right_hip_z
swapped_ref[:, 7] = 0.0    # left_hip_x
swapped_ref[:, 8] = 0.0    # left_hip_z
```

### Joint Weights for Imitation
```python
joint_weights = [
    0.5, 0.3, 0.3,       # abdomen (low priority)
    0.2, 0.8, 2.0, 1.5,  # right leg: hip_x LOW, hip_z MED, hip_y HIGH, knee HIGH
    0.2, 0.8, 2.0, 1.5,  # left leg: same
]
```

---

## 📁 Output Files

### Visualizations
- `Results/humanoid_legs_only_v20.html` - Interactive 3D viewer
- `Results/walking_analysis_v20.png` - Trajectory plots
- `Results/walking_3d_v20.png` - 3D trajectory

### Exports
- `Results/walking_v20.mp4` - Video (640x480)
- `Results/walking_v20.bvh` - BVH animation (Blender/Unity/Maya)
- `Results/walking_v20.json` - JSON for web renderers

---

## 💡 Key Insights

1. **Absolute Reference > Relative Reference**
   - Gyroscope reward (world-up vector) works better than quaternion-based posture reward
   - "Chest pointing at sky" is an absolute, unambiguous reference

2. **Termination > Penalty for Extreme Cases**
   - Death by tilt (termination) handles extreme cases
   - Gradient-based penalties handle fine-tuning
   - Don't need huge penalty weights if you terminate bad episodes

3. **Training Time Sweet Spot**
   - 30M steps is enough for good walking
   - 500 episode length captures 2-3 gait cycles
   - More isn't always better - can lead to "gaming"

4. **Balance Survival vs Style**
   - Too much survival pressure → frozen/conservative behavior
   - Need enough episode length to learn, not so much that survival dominates

---

## 🚀 Next Steps (v22+)

1. **Knee Bend Reward** - Penalize straight knees during swing phase
   ```python
   knee_bend_weight = 4.0
   min_swing_knee_bend = 0.2  # radians (~11°)
   
   # Only apply during swing phase (foot off ground)
   left_in_swing = jp.where(left_foot_z > 0.08, 1.0, 0.0)
   ```

2. **Curriculum Learning** - Start with short episodes, increase gradually

3. **Higher Resolution Video** - Increase MuJoCo framebuffer size in XML

---

## 📚 References

- Brax Physics: https://github.com/google/brax
- CMU MoCap Database: http://mocap.cs.cmu.edu/
- PPO Algorithm: Schulman et al. 2017
