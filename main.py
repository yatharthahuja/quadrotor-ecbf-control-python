import nonlinear_dynamics
import argparse
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from controls import *

parser = argparse.ArgumentParser(
    description='3D Quadcopter linear controller simulation')
parser.add_argument(
    '-T',
    type=float,
    help='Total simulation time',
    default=5.0)
parser.add_argument(
    '--time_step',
    type=float,
    help='Time step simulation',
    default=0.1)
parser.add_argument(
    '-w', '--waypoints', type=float, nargs='+', action='append',
    help='Waypoints')
parser.add_argument('--seed', help='seed', type=int, default=1024)
args = parser.parse_args()

np.random.seed(args.seed)

# The control can be done in a decentralized style
# The linearized system is divided into four decoupled subsystems

# X-subsystem
# The state variables are x, dot_x, pitch, dot_pitch
Ax = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, g, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0]])
Bx = np.array(
    [[0.0],
     [0.0],
     [0.0],
     [1 / Ix]])

# Y-subsystem
# The state variables are y, dot_y, roll, dot_roll
Ay = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, -g, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0]])
By = np.array(
    [[0.0],
     [0.0],
     [0.0],
     [1 / Iy]])

# Z-subsystem
# The state variables are z, dot_z
Az = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]])
Bz = np.array(
    [[0.0],
     [1 / m]])

# Yaw-subsystem
# The state variables are yaw, dot_yaw
Ayaw = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]])
Byaw = np.array(
    [[0.0],
     [1 / Iz]])

# solve LQR: 
Ks = []  # feedback gain matrices K for each subsystem
for A, B in ((Ax, Bx), (Ay, By), (Az, Bz), (Ayaw, Byaw)):
    n = A.shape[0]
    m = B.shape[1]
    Q = np.eye(n)
    Q[0, 0] = 10.  # The first state variable is the one we care about.
    R = np.diag([1., ])
    K, _, _ = lqr(A, B, Q, R)
    Ks.append(K)

# simulation:
# time instants for simulation
t_max = args.T
t = np.arange(0., t_max, args.time_step)


def cl_linear(x, t, u):
    # closed-loop dynamics. u should be a function
    x = np.array(x)
    X, Y, Z, Yaw = x[[0, 1, 8, 9]], x[[2, 3, 6, 7]], x[[4, 5]], x[[10, 11]]
    UZ, UY, UX, UYaw = u(x, t).reshape(-1).tolist()## u input calculated here to calcultae x_dot
    dot_X = Ax.dot(X) + (Bx * UX).reshape(-1)
    dot_Y = Ay.dot(Y) + (By * UY).reshape(-1)
    dot_Z = Az.dot(Z) + (Bz * UZ).reshape(-1)
    dot_Yaw = Ayaw.dot(Yaw) + (Byaw * UYaw).reshape(-1)
    dot_x = np.concatenate(
        [dot_X[[0, 1]], dot_Y[[0, 1]], dot_Z, dot_Y[[2, 3]], dot_X[[2, 3]], dot_Yaw])
    return dot_x


def cl_nonlinear(x, t, u):
    x = np.array(x)
    dot_x = nonlinear_dynamics.f(x, u(x, t) + np.array([m * g, 0, 0, 0]))## u input calculated here to calcultae x_dot
    return dot_x


if args.waypoints:
    # follow waypoints
    signal = np.zeros([len(t), 3])
    num_w = len(args.waypoints)
    for i, w in enumerate(args.waypoints):
        assert len(w) == 3
        signal[len(t) // num_w * i:len(t) // num_w *
               (i + 1), :] = np.array(w).reshape(1, -1)
    X0 = np.zeros(12)
    signalx = signal[:, 0]
    signaly = signal[:, 1]
    signalz = signal[:, 2]
else:
    # Create an random signal to track
    num_dim = 3
    freqs = np.arange(0.1, 1., 0.1)
    weights = np.random.randn(len(freqs), num_dim)  # F x n
    weights = weights / \
        np.sqrt((weights**2).sum(axis=0, keepdims=True))  # F x n
    signal_AC = np.sin(freqs.reshape(1, -1) * t.reshape(-1, 1)
                       ).dot(weights)  # T x F * F x n = T x n
    signal_DC = np.random.randn(num_dim).reshape(1, -1)  # offset
    signal = signal_AC + signal_DC
    signalx = signal[:, 0]
    signaly = signal[:, 1]
    signalz = 0.1 * t
    # initial state
    _X0 = 0.1 * np.random.randn(num_dim) + signal_DC.reshape(-1)
    X0 = np.zeros(12)
    X0[[0, 2, 4]] = _X0

signalyaw = np.zeros_like(signalz)  # we do not care about yaw


def u(x, _t, controller = 'LQR-CBF-QP'):

    # the LQR controller
    if controller == 'LQR':
        dis = _t - t
        dis[dis < 0] = np.inf
        idx = dis.argmin()
        # calculating input values by U = K * err_X
        UX = Ks[0].dot(np.array([signalx[idx], 0, 0, 0]) - x[[0, 1, 8, 9]])[0] 
        UY = Ks[1].dot(np.array([signaly[idx], 0, 0, 0]) - x[[2, 3, 6, 7]])[0]
        UZ = Ks[2].dot(np.array([signalz[idx], 0]) - x[[4, 5]])[0]
        UYaw = Ks[3].dot(np.array([signalyaw[idx], 0]) - x[[10, 11]])[0]
        return np.array([UZ, UY, UX, UYaw])

    elif controller == 'LQR-CBF-QP':
        dis = _t - t
        dis[dis < 0] = np.inf
        idx = dis.argmin()
        # calculating input values by U = K * err_X
        UX = Ks[0].dot(np.array([signalx[idx], 0, 0, 0]) - x[[0, 1, 8, 9]])[0] 
        UY = Ks[1].dot(np.array([signaly[idx], 0, 0, 0]) - x[[2, 3, 6, 7]])[0]
        UZ = Ks[2].dot(np.array([signalz[idx], 0]) - x[[4, 5]])[0]
        UYaw = Ks[3].dot(np.array([signalyaw[idx], 0]) - x[[10, 11]])[0]
        u = np.array([UZ, UY, UX, UYaw])
        u = ecbf_check(u, x, args.time_step)
        return u

    else:
        print("Wrong controller choice!")
            
# simulate
#x_l = odeint(cl_linear, X0, t, args=(u,))
print("Generating Trajectory...")
x_nl = odeint(cl_nonlinear, X0, t, args=(u,))
#print(x_nl[1, 1])
i = 0
print("Trajectory Generated!")
while i < len(x_nl[:]):
    if i == 0:
        min_B = barrier_func(x_nl[i, 0], x_nl[i, 2], x_nl[i, 4])
        min_location = [x_nl[i, 0], x_nl[i, 2], x_nl[i, 4]]
    if barrier_func(x_nl[i, 0], x_nl[i, 2], x_nl[i, 4])<min_B:
        min_B = barrier_func(x_nl[i, 0], x_nl[i, 2], x_nl[i, 4])
        min_location = [x_nl[i, 0], x_nl[i, 2], x_nl[i, 4]]
    if barrier_func(x_nl[i, 0], x_nl[i, 2], x_nl[i, 4]) < 0:
        print(barrier_func(x_nl[i, 0], x_nl[i, 2], x_nl[i, 4]))
    i += 1

print(args.waypoints)
print([[x_nl[len(x_nl)-1, 0], x_nl[len(x_nl)-1, 2], x_nl[len(x_nl)-1, 4]]])
print("Check Done!")
print(min_B)
print(min_location)

######################## plot #######################
fig = plt.figure(figsize=(20, 10))
track = fig.add_subplot(1, 2, 1, projection="3d")
errors = fig.add_subplot(1, 2, 2)

#track.plot(x_l[:, 0], x_l[:, 2], x_l[:, 4], color="r", label="linear")
track.plot(x_nl[:, 0], x_nl[:, 2], x_nl[:, 4], color="g", label="nonlinear")
if args.waypoints:
    for w in args.waypoints:
        track.plot(w[0:1], w[1:2], w[2:3], 'ro', markersize=10.)
else:
    track.text(signalx[0], signaly[0], signalz[0], "start", color='red')
    track.text(signalx[-1], signaly[-1], signalz[-1], "finish", color='red')
    track.plot(signalx, signaly, signalz, color="b", label="command")
track.set_title(
    "Closed Loop response with LQR Controller to random input signal {3D}")
track.set_xlabel('x')
track.set_ylabel('y')
track.set_zlabel('z')
track.legend(loc='lower left', shadow=True, fontsize='small')

#errors.plot(t, signalx - x_l[:, 0], color="r", label='x error (linear)')
#errors.plot(t, signaly - x_l[:, 2], color="g", label='y error (linear)')
#errors.plot(t, signalz - x_l[:, 4], color="b", label='z error (linear)')

errors.plot(t, signalx - x_nl[:, 0], linestyle='-.',
            color='firebrick', label="x error (nonlinear)")
errors.plot(t, signaly - x_nl[:, 2], linestyle='-.',
            color='mediumseagreen', label="y error (nonlinear)")
errors.plot(t, signalz - x_nl[:, 4], linestyle='-.',
            color='royalblue', label="z error (nonlinear)")

errors.plot(t, x_nl[:, 0], linestyle='-.',
            color='r', label="x (nonlinear)")
errors.plot(t, x_nl[:, 2], linestyle='-.',
            color='g', label="y (nonlinear)")
errors.plot(t, x_nl[:, 4], linestyle='-.',
            color='b', label="z (nonlinear)")

errors.set_title("Position error for reference tracking")
errors.set_xlabel("time {s}")
errors.set_ylabel("error")
errors.legend(loc='lower right', shadow=True, fontsize='small')

plt.show()