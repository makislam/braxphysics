# Day 2 Learnings: DeepMimic Humanoid Walking

## 🎯 Goal
Train a Brax humanoid to walk naturally by imitating CMU Motion Capture data using DeepMimic-style reinforcement learning.

---

## 📊 Version History & What Each Fixed

| Version | Problem | Solution | Result |
|---------|---------|----------|--------|
| v1 | Arms horizontal, galloping | - | Arms wrong, galloping motion |
| v2 | Arms wrong position | Set `shoulder2 = 0` (all zeros) | ✅ Arms fixed at sides! |
| v3 | Galloping (both feet same time) | Phase-based gait reward, alternating contacts | ✅ Alternating steps! |
| v4 | Knock-knees | Added knee tracking | ❌ Made it worse (leg crossover) |
| v5-v6 | Crossed legs, drifting | Lateral drift penalty, removed knee tracking | Still crossed legs |
| v7 | Legs crossing through each other | **LEFT/RIGHT LEG SWAP** | ✅ Legs no longer crossing! |
| v8 | Forward tilt persists | Stronger forward tilt penalty | Knees still close |
| v9 | Forward tilt, hunched | **Zero abdomen + strong posture penalties** | ✅ GREAT upright posture! But cowboy stance |
| v10 | Cowboy stance (too wide) | Reduced posture weights + knee params | ❌ Over-corrected (lost good posture) |
| v11 | Cowboy stance | Keep v9 posture, ONLY fix knee params | 🔄 Pending test |

---

## 🔑 Key Discoveries

### 1. MoCap Left/Right Leg Convention is INVERTED vs Brax

**The Problem:**
- CMU MoCap hip_x and hip_z values have opposite sign conventions from Brax
- Result: Robot's legs would cross through each other

**The Solution (v7):**
```python
# Swap left and right leg data, NEGATE hip_x and hip_z
swapped_ref = swapped_ref.at[:, 3].set(-ref_11[:, 7])   # right_hip_x <- -left_hip_x
swapped_ref = swapped_ref.at[:, 4].set(-ref_11[:, 8])   # right_hip_z <- -left_hip_z
swapped_ref = swapped_ref.at[:, 5].set(ref_11[:, 9])    # right_hip_y <- left_hip_y (no negate)
swapped_ref = swapped_ref.at[:, 6].set(ref_11[:, 10])   # right_knee <- left_knee
# ... and vice versa for left leg
```

### 2. MoCap Abdomen Values Cause Forward Tilt

**The Problem:**
- CMU MoCap abdomen_y values are non-zero (forward lean in their coordinate system)
- When applied to Brax, robot bends forward constantly

**The Solution (v9):**
```python
# Zero out ALL abdomen values - don't trust MoCap here
swapped_ref = swapped_ref.at[:, 0].set(0.0)   # abdomen_y = 0
swapped_ref = swapped_ref.at[:, 1].set(0.0)   # abdomen_z = 0
swapped_ref = swapped_ref.at[:, 2].set(0.0)   # abdomen_x = 0
```

### 3. Forcing Hip Abduction Causes Cowboy Stance

**The Problem:**
- v9 set `target_hip_x = 0.05` to push legs apart
- Combined with `hip_abduction_weight = 5.0`, this forced legs too wide

**The Solution (v11):**
```python
# Let robot find natural stance
target_hip_x = 0.0           # Neutral, not forced outward
hip_abduction_weight = 1.0   # Much lower weight

# Also set reference hip_x to zero
swapped_ref = swapped_ref.at[:, 3].set(0.0)   # right_hip_x = 0
swapped_ref = swapped_ref.at[:, 7].set(0.0)   # left_hip_x = 0
```

### 4. Knee Separation Needs BOTH Min AND Max Bounds

**The Problem:**
- Only having `min_knee_separation` led to cowboy stance when set too high
- v9 had `min_knee_separation = 0.20` (20cm) - way too wide

**The Solution (v11):**
```python
# Goldilocks zone - not too close, not too wide
min_knee_separation = 0.08   # 8cm minimum (prevent knock-knees)
max_knee_separation = 0.25   # 25cm maximum (prevent cowboy stance)

# Reward being in the zone
too_close = jp.where(knee_separation < min_knee_separation,
                     jp.square(min_knee_separation - knee_separation), 0.0)
too_far = jp.where(knee_separation > max_knee_separation,
                   jp.square(knee_separation - max_knee_separation), 0.0)
```

---

## ⚠️ Mistakes to Avoid

### 1. Don't Change Multiple Things at Once
When v9 had great posture but wide knees, v10 incorrectly:
- ❌ Reduced `posture_weight` from 15.0 to 8.0
- ❌ Reduced `height_penalty_weight` from 8.0 to 5.0
- ❌ AND changed knee parameters

**Correct approach (v11):**
- ✅ Keep ALL posture parameters from v9
- ✅ ONLY change knee-related parameters

### 2. Don't Trust MoCap Data Blindly
- Hip conventions may be inverted
- Abdomen/torso values may not translate correctly
- Always verify coordinate system mappings

### 3. Don't Over-Constrain
- High weights on multiple competing objectives cause instability
- Let the robot find natural solutions within reasonable bounds

---

## 🏗️ Architecture Notes

### Joint Mapping (Legs Only - 11 DOFs)
```
Index 0-2:  abdomen_y, abdomen_z, abdomen_x
Index 3-6:  right_hip_x, right_hip_z, right_hip_y, right_knee
Index 7-10: left_hip_x, left_hip_z, left_hip_y, left_knee
```

### Arms Fixed At
```python
_fixed_arm_positions = [0, 0, 0, 0, 0, 0]  # All zeros = arms at sides
```

### Reference Motion Processing Pipeline
```
1. Load CMU MoCap (ASF/AMC files)
2. Extract joint angles with coordinate conversion
3. Average across 6 subjects (for robustness)
4. Resample from 120 FPS → 67 FPS (Brax dt=0.015)
5. Apply leg swap fix (negate hip_x, hip_z)
6. Zero out abdomen values
7. Set hip_x to neutral (0.0)
```

---

## 📈 Reward Structure (v11)

| Reward | Weight | Purpose |
|--------|--------|---------|
| Healthy (alive) | 5.0 | Survival priority |
| Pose matching | 2.0 | Match MoCap joint angles |
| Forward progress | 2.0 | Walk at ~1 m/s |
| Posture | 15.0 | Stay upright |
| Root position | 2.0 | Keep torso centered |
| Height penalty | 8.0 | Maintain 1.3m height |
| Gait (alternating) | 2.0 | Proper foot timing |
| Knee separation | 3.0 | 8-25cm width zone |
| Lateral penalty | 3.0 | Don't drift sideways |
| Hip abduction | 1.0 | Neutral hip angle |
| Velocity matching | 0.3 | Match joint velocities |
| Control cost | 0.02 | Smooth actuations |
| Foot skate | 0.3 | No sliding feet |

---

## 🔬 Debugging Techniques

### 1. Visual Inspection
- Watch for specific artifacts: crossed legs, forward lean, knock-knees, cowboy stance
- Note which body parts move correctly vs incorrectly

### 2. Phase-Based Analysis
- Check if issues correlate with walking phase (0-0.5 vs 0.5-1.0)
- Left/right asymmetries often indicate coordinate system issues

### 3. Isolate Variables
- Simplify to legs-only first
- Get one thing working before adding complexity
- When fixing bugs, change ONE parameter at a time

### 4. Reference Motion Verification
- Print joint angle ranges from MoCap data
- Verify signs and magnitudes make physical sense
- Compare left vs right leg values

---

## 🎯 What's Working (After v9/v11)

- ✅ Arms fixed naturally at sides
- ✅ Alternating foot contacts (no galloping)
- ✅ Legs don't cross through each other
- ✅ Upright torso posture
- ✅ Maintains ~1.3m height
- ✅ Walks forward at reasonable speed
- ✅ Minimal lateral drift

## 🔄 Still Being Refined

- 🔄 Knee width (v11 should fix cowboy stance)
- 🔄 Arm swing (disabled, can re-enable later)
- 🔄 True DeepMimic (sample from subjects instead of averaging)

---

## 💡 Future Improvements

1. **Re-enable arm swing** once legs are perfect
2. **True DeepMimic**: Sample randomly from motion clips instead of averaging
3. **More motion clips**: Add running, turning, stairs
4. **Domain randomization**: Vary physics parameters for robustness
5. **Curriculum learning**: Start easy, increase difficulty

---

## 📁 Files

- `humanoid_mocap.ipynb` - Main training notebook
- `Training_Data/` - CMU MoCap ASF/AMC files
- `humanoid_legs_only.html` - Visualization output
- `day2_learnings.md` - This file

---

*Last updated: Day 2 of training*
