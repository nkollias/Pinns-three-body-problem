import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ Define the 3-body system (planar, equal masses)
# ------------------------------------------------------------
G = 1.0
m1 = m2 = m3 = 1.0

def three_body_rhs(t, y):
    # y = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3]
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = y

    # pairwise position vectors
    r12 = np.array([x2 - x1, y2 - y1])
    r13 = np.array([x3 - x1, y3 - y1])
    r23 = np.array([x3 - x2, y3 - y2])

    # distances cubed
    r12_3 = np.linalg.norm(r12)**3
    r13_3 = np.linalg.norm(r13)**3
    r23_3 = np.linalg.norm(r23)**3

    # accelerations
    a1 = G * (m2 * r12 / r12_3 + m3 * r13 / r13_3)
    a2 = G * (m1 * (-r12) / r12_3 + m3 * r23 / r23_3)
    a3 = G * (m1 * (-r13) / r13_3 + m2 * (-r23) / r23_3)

    return [vx1, vy1, a1[0], a1[1],
            vx2, vy2, a2[0], a2[1],
            vx3, vy3, a3[0], a3[1]]

# ------------------------------------------------------------
# 2️⃣ Figure-eight initial conditions (Chenciner & Montgomery 2000)
# ------------------------------------------------------------
y0 = np.array([
    -0.97000436,  0.24308753,  0.466203685,  0.43236573,
     0.97000436, -0.24308753,  0.466203685,  0.43236573,
     0.0,         0.0,        -0.93240737,  -0.86473146
])

# Integration time
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 5000)

# ------------------------------------------------------------
# 3️⃣ Integrate with SciPy
# ------------------------------------------------------------
sol = solve_ivp(three_body_rhs, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9, method="DOP853")

t = sol.t
x1, y1, x2, y2, x3, y3 = sol.y[0], sol.y[1], sol.y[4], sol.y[5], sol.y[8], sol.y[9]

# ------------------------------------------------------------
# 4️⃣ Estimate period by autocorrelation
# ------------------------------------------------------------
# Use one coordinate (e.g., x1) to detect periodicity
x1_centered = x1 - np.mean(x1)
corr = np.correlate(x1_centered, x1_centered, mode='full')
corr = corr[corr.size // 2:]

# Find the first local maximum after lag=0
from scipy.signal import find_peaks
peaks, _ = find_peaks(corr, distance=len(corr)//20)
if len(peaks) > 0:
    T_est = t[peaks[0]]
else:
    T_est = np.nan

print(f"Estimated period ≈ {T_est:.4f} time units")

# ------------------------------------------------------------
# 5️⃣ Visualization
# ------------------------------------------------------------
plt.figure(figsize=(6,6))
plt.plot(x1, y1, label="Body 1")
plt.plot(x2, y2, label="Body 2")
plt.plot(x3, y3, label="Body 3")
plt.scatter([x1[0], x2[0], x3[0]], [y1[0], y2[0], y3[0]], color="red", label="Start")
plt.gca().set_aspect("equal", "box")
plt.legend()
plt.title("Three-Body Figure-Eight Orbit")
plt.show()

plt.figure()
plt.plot(t, x1)
plt.title("x₁(t) — used for period estimation")
plt.xlabel("Time")
plt.ylabel("x₁")
plt.show()
