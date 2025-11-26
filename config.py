import numpy as np

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================
DT = 0.01               # 10ms Time step (Optimized for animation speed)
TMAX = 60.0             # Duration
G = 9.81                # Gravity

# ============================================================================
# THREAT SETTINGS (Ranges for Random Generation)
# ============================================================================
# The target will pick random values within these bounds
TARGET_SPEED_MIN = 900  # m/s
TARGET_SPEED_MAX = 1300 # m/s

# Start Position Bounds (Raised altitude to prevent early ground crashes)
X_START_MIN = 35000     
X_START_MAX = 45000
Y_START_MIN = -15000    
Y_START_MAX = 15000
Z_START_MIN = 25000     # Higher start = longer flight time for intercept
Z_START_MAX = 35000

# Maneuver Randomness
MANEUVER_CHANCE = 0.02  # Chance per frame to start a maneuver

# ============================================================================
# RADAR SETTINGS (AN/TPY-2 Simulation)
# ============================================================================
RADAR_RANGE = 60000.0   # Meters (Increased range)
RADAR_NOISE_STD = 15.0  # Meters 
TRACK_UPDATE_RATE = 0.05 

# ============================================================================
# INTERCEPTOR (THAAD) SETTINGS
# ============================================================================
MISSILE_VELOCITY = 2800        # m/s
MISSILE_START_POS = np.array([0, 0, 0])
LAUNCH_DELAY = 1.0             # Seconds

# Guidance
GUIDANCE_GAIN = 5.0     # Increased aggressiveness
MAX_G_LOAD = 40.0 * G   
KILL_DISTANCE = 8.0     # Slightly larger hit box