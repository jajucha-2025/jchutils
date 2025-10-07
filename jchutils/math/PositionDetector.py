import math

def pd_euler(state, v, delta, L, dt):
    """
    Euler integration single-step.
    state: (x, y, theta)
    v: rear-wheel linear speed (m/s)
    delta: steering angle (rad)
    L: wheelbase (m)
    dt: time step (s)
    returns: new_state (x, y, theta)
    """
    x, y, theta = state
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta += (v / L) * math.tan(delta) * dt
    # normalize theta to [-pi, pi) if desired
    theta = (theta + math.pi) % (2*math.pi) - math.pi
    return (x, y, theta)


def _deriv(state, v, delta, L):
    x, y, theta = state
    dx = v * math.cos(theta)
    dy = v * math.sin(theta)
    dtheta = (v / L) * math.tan(delta)
    return dx, dy, dtheta


def pd_rk2(state, v, delta, L, dt):
    x, y, theta = state
    k1 = _deriv((x, y, theta), v, delta, L)
    k2 = _deriv((x + 0.5*dt*k1[0], y + 0.5*dt*k1[1], theta + 0.5*dt*k1[2]), v, delta, L)
    x_next = x + dt*k2[0]
    y_next = y + dt*k2[1]
    theta_next = theta + dt*k2[2]
    theta_next = (theta_next + math.pi) % (2*math.pi) - math.pi
    return (x_next, y_next, theta_next)

def pd_rk4(state, v, delta, L, dt):
    """
    One RK4 step for better accuracy.
    """
    x, y, theta = state

    k1 = _deriv((x, y, theta), v, delta, L)
    k2 = _deriv((x + 0.5*dt*k1[0], y + 0.5*dt*k1[1], theta + 0.5*dt*k1[2]), v, delta, L)
    k3 = _deriv((x + 0.5*dt*k2[0], y + 0.5*dt*k2[1], theta + 0.5*dt*k2[2]), v, delta, L)
    k4 = _deriv((x + dt*k3[0], y + dt*k3[1], theta + dt*k3[2]), v, delta, L)

    x_next = x + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    y_next = y + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    theta_next = theta + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

    theta_next = (theta_next + math.pi) % (2*math.pi) - math.pi
    return (x_next, y_next, theta_next)

def main():
    L = 0.25  # wheelbase
    dt = 0.02 # delta time
    state = (0.0, 0.0, 0.0)  # start pose (x, y, theta)
    v = 1.0 # speed
    delta_deg = 10.0 # steer (deg)
    delta_rad = math.radians(delta_deg)

    for i in range(200):  
        state = pd_rk4(state, v, delta_rad, L, dt)
        if i % 50 == 0:
            print(f"t={i*dt:.2f}s  x={state[0]:.3f} y={state[1]:.3f} theta={math.degrees(state[2]):.2f}deg")

if __name__ == "__main__":
    main()