import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- Constants and setup ---
G = 1.0  # gravitational constant (dimensionless units)
# --- Parameters ---
m1 = m2 = m3 = 1.0
endTime=2

def three_body_equations(t, y, m1, m2, m3):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = y
     # Distances
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    # Accelerations (Newton’s law)
    ax1 = G * (m2*(x2 - x1)/r12**3 + m3*(x3 - x1)/r13**3)
    ay1 = G * (m2*(y2 - y1)/r12**3 + m3*(y3 - y1)/r13**3)
    ax2 = G * (m1*(x1 - x2)/r12**3 + m3*(x3 - x2)/r23**3)
    ay2 = G * (m1*(y1 - y2)/r12**3 + m3*(y3 - y2)/r23**3)
    ax3 = G * (m1*(x1 - x3)/r13**3 + m2*(x2 - x3)/r23**3)
    ay3 = G * (m1*(y1 - y3)/r13**3 + m2*(y2 - y3)/r23**3)

    return [vx1, vy1, ax1, ay1,
            vx2, vy2, ax2, ay2,
            vx3, vy3, ax3, ay3]


# Initial positions and velocities
#body 1    x1,y1,vx1,vy1
#body 2    x2,y2,vx2,vy2
#body 3    x3,y3,vx3,vy3
omega = 0.3
# ====  Lagrange   ========== 
y0 =[

 -0.062469288918794566  , -0.2899597451750707  , -0.20031387041683557  , 0.7621937147909932,
    0.19927554190827557 ,  -0.04502451106291724  , 1.5529931234041516 ,  -0.1923023859303955,
    -0.11608888319571337 ,  0.33293738581061555  , -1.3926241410664213 ,  -0.5648285658383415,

]


# Time grid
t_span = (0, endTime)
t_eval = np.linspace(*t_span, 1000)

# --- Solve the system ---
sol = solve_ivp(three_body_equations, t_span, y0, args=(m1, m2, m3),
                t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Extract trajectories
x1, y1 = sol.y[0], sol.y[1]
x2, y2 = sol.y[4], sol.y[5]
x3, y3 = sol.y[8], sol.y[9]

# --- Animation setup ---
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_title("Three-Body Problem Animation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.axis("equal")
ax.grid(True)

# Plot trails (will be updated each frame)
trail_len = 1000  # number of previous positions to show as tail
line1, = ax.plot([], [], 'b-', lw=1)
line2, = ax.plot([], [], 'orange', lw=1)
line3, = ax.plot([], [], 'g-', lw=1)

# Plot moving bodies
dot1, = ax.plot([], [], 'o', color='tab:blue', markersize=8)
dot2, = ax.plot([], [], 'o', color='tab:orange', markersize=8)
dot3, = ax.plot([], [], 'o', color='tab:green', markersize=8)

# Mark starting positions
ax.scatter(x1[0], y1[0], color='tab:blue', s=100, edgecolor='k', label='Start Body 1')
ax.scatter(x2[0], y2[0], color='tab:orange', s=100, edgecolor='k', label='Start Body 2')
ax.scatter(x3[0], y3[0], color='tab:green', s=100, edgecolor='k', label='Start Body 3')

ax.legend(loc='upper right')

# --- Update function for animation ---
def update(frame):
    i = frame
    i_min = max(0, i - trail_len)

    # update trails
    line1.set_data(x1[i_min:i], y1[i_min:i])
    line2.set_data(x2[i_min:i], y2[i_min:i])
    line3.set_data(x3[i_min:i], y3[i_min:i])

    # update body positions (wrap scalars in lists!)
    dot1.set_data([x1[i]], [y1[i]])
    dot2.set_data([x2[i]], [y2[i]])
    dot3.set_data([x3[i]], [y3[i]])

    return line1, line2, line3, dot1, dot2, dot3


# --- Create animation ---
ani = FuncAnimation(fig, update, frames=len(t_eval), interval=endTime, blit=True)

plt.show()
