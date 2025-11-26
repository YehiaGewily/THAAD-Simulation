# THAAD-Like Ballistic Missile Defense Simulation

A Python-based study of Guidance, Navigation, and Control (GNC) for kinetic interception

---

## 📖 Overview

This project is a 3D kinematic simulation of a **THAAD-style terminal missile defense system**, built in Python using NumPy, Matplotlib, and Rich.

The goal is **not** to model any real weapon system, but to:

* Deconstruct the maths and physics behind **"hitting a bullet with another bullet"**.
* Explore core **Guidance, Navigation, and Control (GNC)** concepts.
* Visualize how a defensive interceptor reacts to an incoming ballistic threat under noisy sensing, drag, gravity, and staging.

Instead of being just an animation, the code closes a simplified but complete **GNC loop**:

1. **Detection** – radar-like noisy measurements of a ballistic threat in 3D.
2. **Estimation** – a 6‑state Kalman Filter estimating true position and velocity.
3. **Decision** – fire-control logic determining *if and when* to launch based on predicted impact.
4. **Guidance** – Augmented Proportional Navigation (APN) steering the interceptor.
5. **Physics** – multi-stage interceptor dynamics, mass depletion, drag, and gravity.
6. **Engagement Assessment** – continuous collision detection (CPA) and a battle-damage summary.

All of this is visualized in an interactive **3D Matplotlib plot** with a **time slider** and a **Rich-based BDA table** in the terminal.

> ⚠️ This is an educational / hobby simulation. Numbers are illustrative, not real-world data.

---

## 📂 Project Structure

From the repository root:

```text
.
├── THAAD MAX_performance prototype.py   # v5.0 – "Max-Performance" / High-Margin Mode
├── THAAD Ultimlate Prototype.py         # v5.1 – "Engineering Mode" / Tuned Realism
├── Older Attempts/                      # Scratch & learning experiments
│   └── ... (early prototypes, tests, etc.)
└── README.md
```

### Main simulation files

#### `THAAD Ultimlate Prototype.py` – Engineering Mode (Tuned Realism)

This is the **primary** simulation to run.

* Uses randomized threat geometry (altitude, range, azimuth) aimed near the defended origin.
* Simulates a two-stage interceptor with realistic-ish thrust and G-limits.
* Applies a 6-state Kalman Filter with gravity bias to track the threat.
* Computes a **Predicted Impact Point (PIP)** and launches only when:

  * The threat is clearly descending.
  * The track has been observed long enough to be stable.
  * The PIP lies within a plausible defended area.
* Uses continuous collision detection and a relatively tight kill radius.

Result: **interception is likely but not guaranteed.** Some random scenarios will be outside the interceptor's kinematic envelope, which is exactly the point of an engineering-style sim.

#### `THAAD MAX_performance prototype.py` – Max-Performance Intercept Demo

This mode shares the same architecture (Kalman, PN guidance, CPA, 3D replay), but with more aggressive tuning:

* Slightly easier threat profiles (closer / lower, with mild drag compensation).
* Stronger interceptor (higher thrust, longer burn, higher G-limit, larger kill radius).
* Launches as soon as the track is confirmed and the threat is descending.

Result: **high probability of intercept**, visually satisfying trajectories, and big closing velocities. Use this mode for demonstrations or when you want a near-guaranteed hit to study the geometry.

### Older Attempts / Scratch Files

The `Older Attempts/` folder and early scripts (e.g. `THAAD Simulation Prototype.py`) are **scratch and learning experiments**. They were used to:

* Test basic ballistic motion and drag.
* Prototype simple PN interceptors.
* Play with Matplotlib 3D visualization and animation.

They are kept for historical context and quick reference, but the main work now happens in:

* `THAAD Ultimlate Prototype.py`
* `THAAD MAX_performance prototype.py`

---

## 🚀 Quick Start

### 1. Install requirements

You need Python 3.9+ and the following packages:

```bash
pip install numpy matplotlib rich
```

On some systems you may also need `tkinter` (for the `TkAgg` backend) if you want a separate interactive window.

### 2. Run the engineering mode (recommended)

```bash
python "THAAD Ultimlate Prototype.py"
```

### 3. Run the max-performance demo

```bash
python "THAAD MAX_performance prototype.py"
```

### 4. Controls

* **Mouse drag** – rotate the 3D scene.
* **Mouse wheel / trackpad** – zoom in and out.
* **Time slider (bottom)** – scrub through the engagement frame-by-frame.

In the terminal you’ll see Rich logs with launch events, kill / fail messages, and a final BDA table.

---

## 🧠 How the Simulation Works

### Coordinate System & Units

* Right-handed Cartesian coordinates:

  * **x** – East (meters)
  * **y** – North (meters)
  * **z** – Altitude above defended ground plane (meters)
* **z = 0** represents ground level / defended plane.
* Time is in **seconds**; velocities in **m/s**; accelerations in **m/s²**.
* Gravity is constant: `g = 9.81 m/s²`.

Earth curvature, rotation, and detailed atmosphere models are ignored, which is acceptable for terminal engagements on the order of tens of kilometers.

---

### Radar Model & Kalman Filtering

The radar is modeled as a noisy 3D sensor:

* True position: **p = (x, y, z)**
* True velocity: **v = (vx, vy, vz)**
* Measurements:

> p_meas = p + N(0, σp²)
>
> v_meas = v + N(0, σv²)

A **linear Kalman Filter** estimates the state

> x = [x, y, z, vx, vy, vz]^T

Key points:

* **State transition**: constant-velocity model with position integrating velocity each time step.
* **Measurement model**: identity matrix; we directly observe position and velocity, but noisy.
* **Process noise**: small diagonal Q to keep the filter flexible.
* **Measurement noise**: R built from the chosen position and velocity standard deviations.

To help the filter understand that the threat is falling, the prediction step includes a **gravity bias** on vertical velocity:

```python
self.x[5] -= g * dt
```

This reduces vertical lag and gives more realistic predicted impact points.

The filter also exposes a helper like:

```python
def predict_impact(self, g):
    # solve z(t) = 0 assuming ballistic motion
```

which analytically solves for the time when z = 0 and predicts the corresponding ground impact point.

---

### Threat Model – Ballistic Target

The incoming warhead is approximated as a **point mass** with gravity and a simple drag term:

`a_threat = [0, 0, -g] - C_d * ||v|| * v`

Each run spawns a new random scenario:

* Initial altitude z0 in the tens of kilometers (e.g. 50–70 km).
* Initial ground range r0 tens of kilometers away from the origin.
* Random azimuth angle so the threat can arrive from any direction.

To ensure it actually aims near the defended origin, the code:

1. Chooses a nominal time-to-impact T_imp.
2. Solves the 1D vertical ballistic equation (constant gravity, no drag) for v_z0 such that z(T_imp) ≈ 0.
3. Solves for v_x0 and v_y0 so that after T_imp the footprint is roughly at (0, 0).
4. Multiplies the whole velocity vector by a small drag compensation factor so that with drag, the trajectory still ends near the defended region.

This produces a variety of realistic-looking terminal threat trajectories.

---

### Interceptor Model – Two-Stage Vehicle

The interceptor is modeled as a two-stage vehicle:

1. **BOOST stage**

   * Time-limited rocket burn (`boost_time`).
   * Thrust magnitude (`boost_thrust`).
   * Mass decreases linearly from `mass_wet` to `mass_dry`.
   * Drag uses a "booster" coefficient.
   * Maneuver commands are partially applied (e.g. 10%) on top of thrust to mimic limited thrust-vectoring.

2. **KV_COAST stage**

   * After burnout, the booster is gone; mass is set to `kv_mass`.
   * No main thrust (pure coast).
   * Drag uses a separate smaller `kv_drag` coefficient.
   * Maneuver commands represent kill-vehicle divert thrusters.

Internally, the forces on the interceptor are:

> F_total = F_thrust + F_grav + F_drag + F_maneuver

and acceleration is a = F_total / m. Mass is updated during boost to reflect fuel burn.

---

### Guidance – Augmented Proportional Navigation (APN)

The interceptor uses a classic **Proportional Navigation (PN)** guidance law with gravity compensation.

1- Compute relative vectors:

> r = p_target - p_interceptor
>
> v = v_target - v_interceptor

2- Line-of-sight (LOS) rotation rate:

> omega = (r × v) / (|r|² + ε)

3- PN acceleration command:

> a_PN = N * (v × omega)

4- Augment with gravity bias:

> a_cmd = a_PN + [0, 0, g]

5- Enforce a **G-limit**:

> |a_cmd| ≤ max_g * g

If the commanded acceleration exceeds the allowed limit, it is scaled back. The final command is logged as a G-load history for post-mission analysis.

Intuition:

* If the LOS is not rotating, the interceptor is already on a collision course.
* LOS rotation triggers lateral acceleration to drive LOS rate back toward zero.

---

### Continuous Collision Detection (CPA)

At typical closing speeds (several km/s), the threat and interceptor can move **tens of meters per time step**. A naive check like:

```python
if np.linalg.norm(threat.pos - interceptor.pos) < kill_radius:
    # hit
```

would miss collisions because the objects may "jump over" each other between frames.

To fix this, the simulation uses **Closest Point of Approach (CPA)** over each time step:

1 Let Δp = p2 - p1, and Δv = v2 - v1.
2- The time of minimum separation (assuming linear motion over the step) is:

> t_min = - (Δp · Δv) / |Δv|²

3- If 0 ≤ t_min ≤ dt, the distance at that instant is checked; otherwise the distances at the start and end of the step are compared.

If the minimum distance within the step is smaller than `kill_radius`, the sim:

* Marks a **HIT** outcome.
* Computes the precise collision time and positions.
* Places a bright explosion marker at that point in 3D.

---

### Battle Damage Assessment (BDA)

At the end of each run, the terminal prints a **Rich** table summarizing the engagement, for example:

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ BATTLE DAMAGE ASSESSMENT ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Outcome      │ KILL       │
│ Kill Alt     │ 42.5 km    │
│ Miss Dist    │ 0.12 m     │
│ Closing Vel  │ 3100 m/s   │
│ Max G-Load   │ 12.4 G     │
└──────────────────────────┘
```

Possible outcomes:

* `KILL` – interceptor successfully hits the threat within the kill radius.
* `GROUND` – threat impacts the ground (z = 0) before intercept.
* `TIMEOUT` – simulation time ends while the threat is still airborne and not yet killed.

The BDA values and the 3D replay complement each other: you get both **numbers** and **geometry**.

---

## 🎮 Modes Summary

| Mode / File                          | Style            | Intercept Guarantee | Notes                                                                                   |
| ------------------------------------ | ---------------- | ------------------- | --------------------------------------------------------------------------------------- |
| `THAAD Ultimlate Prototype.py`       | Engineering Mode | Not guaranteed      | Realistic-ish thrust, drag, G-limits, and launch logic based on predicted impact point. |
| `THAAD MAX_performance prototype.py` | Max-Performance  | Very high           | Strong interceptor, slightly easier threat, good for demos and intuition.               |

Both modes share the same core ideas: Kalman tracking, APN guidance, continuous collision detection, and 3D visualization.

---

## ⚠️ Limitations & Assumptions

This is a **learning tool**, not a design or validation environment. Major simplifications include:

* **Aerodynamics** – drag is a simple constant-coefficient model; no Mach dependence, no atmosphere layering.
* **Earth Model** – flat Earth with constant gravity; no curvature, rotation, or Coriolis effects.
* **Sensors** – radar is treated as a perfect 3D position/velocity sensor with Gaussian noise; real systems work in range/angle space, with clutter, dropouts, and fusion from multiple sensors.
* **Guidance** – classic PN with a simple gravity bias; real guidance laws include many more constraints and details.
* **No classification** – all parameters are chosen for nice behavior and readability, not based on any sensitive or proprietary data.

Treat this repository as a **sandbox to build intuition** about missile defense kinematics, not a predictive model for any real system.

---

## 🔮 Possible Extensions

Some ideas for future work:

* Multiple simultaneous threats and interceptors.
* Different guidance laws (pure pursuit, augmented PN variants, impact-angle constraints).
* More realistic radar models (range/azimuth/elevation, update rate, clutter, missed detections).
* Batch Monte Carlo runs to gather statistics on probability of kill vs. configuration.
* Simple GUI front-end to let users configure scenarios without editing code.

---

## 📜 License

This project is intended for educational and research purposes. If you plan to release it publicly, a good default choice is the **MIT License**; you can add a `LICENSE` file with the standard MIT text if you’d like it to be formally open source.
