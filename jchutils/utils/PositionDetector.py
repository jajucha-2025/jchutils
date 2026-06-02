import math


### -----------------------------------
### v: rear-wheel linear speed (cm/s)
### deltaL: steering angle - Left (rad)
### deltaR: steering angle - Right (rad)
### dt: time step (s)
### 
### L은 불변값이므로 _car에 추가 요함
### -----------------------------------

### -----------------------------------
### =========================
### vehicle params
### =========================
### state: (x, y, theta)
###
### 전부 _car에 속성 추가 요함 (현재 임시)
### L: wheelbase (cm) = 18.0
### T: track (cm) = 15.0
### KPO: kingpin offset (cm) = 1.4 
### LR: CG (cm) = 9.0
### -----------------------------------

TRACK = 15.0
WHEELBASE = 18.0
KINGPIN_OFFSET = 1.4 
LR = WHEELBASE * 0.5


def normalize_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def equivalent_steer(deltaL, deltaR):
    tanL = math.tan(deltaL)
    tanR = math.tan(deltaR)
    
    if abs(tanL + tanR) < 1e-9:
        return 0.0
    
    tan_eq = (2.0 * tanL * tanR) / (tanL + tanR)
    
    return math.atan(tan_eq)


def _deriv(state, v, deltaL, deltaR, L):
    x, y, theta = state

    delta_eq = equivalent_steer(deltaL, deltaR)
    
    tan_delta = math.tan(delta_eq)
    
    beta = math.atan((LR / L) * tan_delta)
    
    denom = L - KINGPIN_OFFSET * tan_delta
    
    if abs(denom) < 1e-6:
        omega = 0.0
    else:
        omega = (v * tan_delta) / denom
    
    heading = theta + beta + math.pi / 2
    
    dx = v * math.cos(heading)
    dy = v * math.sin(heading)
    dtheta = omega
    
    return dx, dy, dtheta


def pd_euler(state, v, deltaL, deltaR, L, dt):
    x, y, theta = state

    dx, dy, dtheta = _deriv(
        state,
        v,
        deltaL,
        deltaR,
        L
    )

    x += dx * dt
    y += dy * dt
    theta += dtheta * dt

    theta = normalize_angle(theta)

    return (x, y, theta)


def pd_rk2(state, v, deltaL, deltaR, L, dt):
    x, y, theta = state
    k1 = _deriv(
        (x, y, theta),
        v, deltaL, deltaR, L
    )
    k2 = _deriv(
        (
            x + 0.5 * dt * k1[0],
            y + 0.5 * dt * k1[1],
            theta + 0.5 * dt * k1[2]
        ),
        v, deltaL, deltaR, L
    )
    
    x_next = x + dt * k2[0]
    y_next = y + dt * k2[1]
    theta_next = theta + dt * k2[2]

    theta_next = normalize_angle(theta_next)

    return (x_next, y_next, theta_next)


def pd_rk4(state, v, deltaL, deltaR, L, dt):
    x, y, theta = state
    
    k1 = _deriv(
        (x, y, theta),
        v, deltaL, deltaR, L
    )
    k2 = _deriv(
        (
            x + 0.5 * dt * k1[0],
            y + 0.5 * dt * k1[1],
            theta + 0.5 * dt * k1[2]
        ),
        v, deltaL, deltaR, L
    )
    k3 = _deriv(
        (
            x + 0.5 * dt * k2[0],
            y + 0.5 * dt * k2[1],
            theta + 0.5 * dt * k2[2]
        ),
        v, deltaL, deltaR, L
    )
    k4 = _deriv(
        (
            x + dt * k3[0],
            y + dt * k3[1],
            theta + dt * k3[2]
        ),
        v, deltaL, deltaR, L
    )

    x_next = x + (dt / 6.0) * (
        k1[0] + 2*k2[0] + 2*k3[0] + k4[0]
    )
    y_next = y + (dt / 6.0) * (
        k1[1] + 2*k2[1] + 2*k3[1] + k4[1]
    )
    theta_next = theta + (dt / 6.0) * (
        k1[2] + 2*k2[2] + 2*k3[2] + k4[2]
    )
    
    theta_next = normalize_angle(theta_next)
    
    return (x_next, y_next, theta_next)


