import nonlinear_dynamics
import argparse
import numpy as np
import scipy
from scipy.integrate import odeint
from nonlinear_dynamics import g, m, Ix, Iy, Iz
import sympy
from sympy.diffgeom.rn import R2_r, R2_p
from sympy.diffgeom import (LieDerivative, TensorProduct) 
import quadprog


## POSSIBLE MISTAKES:
# 1. f(x), g(x) calculation: correct indices and dimensions, corresponding to accurate state variable
# 2. dBdX function: indices and accuracy of gradient

def quadprog_solve_qp(P, q, G, h, A=None, b=None):
    # minimise for x: (1/2)*xT*P*x + qT*x (x is a column vector)
    # constraints: G*x <= h ; A*x = b
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -1*h
        meq = 0
    #print(type(qp_b))
    #print(type(qp_C))
    #print(type(qp_a))
    #print(type(qp_G))
    #print(qp_b.shape)
    #print(qp_C.shape)
    #print(qp_a.shape)
    #print(qp_G.shape)
    #print(qp_b)
    #print(qp_C)
    #print(qp_a)
    #print(qp_G)
    #print(meq)
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def cl_nonlinear(x, t, u):
    x = np.array(x)
    dot_x = nonlinear_dynamics.f(x, u(x, t) + np.array([m * g, 0, 0, 0]))## u input calculated here to calcultae x_dot
    return dot_x

# For LieDerivative example=> https://docs.sympy.org/latest/modules/diffgeom.html
# Possible LieDerivative help=> https://pycontroltools.readthedocs.io/en/latest/pycontroltools.html
def lie_derivative(f, v):
    # Lfh : first-order lie derivative of f wrt h
    # Lfh(x) = dhdx(x) * f(x)
    return np.dot(f, v)#LvS # Lie Derivative of Scalar Field 'S' wrt to Vector Field 'v' 

# need to implement a function which returns Barrier function with location as argument: def barrier_func(x, y, z):return B
def barrier_func(x, y, z):
    # constraint: stay outside a circle of radius 0.5 at (1,1,0)
    # constraint formulation(Barrier Function): 
    #B = {}(Cuboidal)
    B = (x-1)**2 + (y-1)**2 - 0.25 # (Cylindrical: circle in XY plane, Z value immaterial)
    return B #1x1

def dBdx(x, y, z):
    gradB = np.array([[2*x-2],[0],[2*y-2],[0],[0],[0],[0],[0],[0],[0],[0],[0]]).T # 1x12
    # returns d(B)/dX for Lie Derivative calculation
    return gradB[0]#2*x + 2*y -4

def dBdX(X, Y, Z, r = 1):
    #print(r)
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    B = (x-1)**2 + (y-1)**2 - 0.25
    if r>1:
        B = x**2 + y**2 # effective equation for easier differential since constants go zero
    if r>2:
        return np.zeros(12)
    else:   
        gradB = barrier_func(X, Y, Z)*np.ones(12).T # 1x12
        i = 0 
        while i < r:
            gradB = np.array([[B.diff(x)],[0],[B.diff(y)],[0],[0],[0],[0],[0],[0],[0],[0],[0]]).T # 1x12
            i += 1
        # returns d(B)/dX for Lie Derivative calculation
        gradB[0][0] = sympy.lambdify(x, gradB[0][0])
        gradB[0][0] = gradB[0][0](X)
        gradB[0][2] = sympy.lambdify(y, gradB[0][2])
        gradB[0][2] = gradB[0][2](Y)
        return gradB[0]

def nu_B(X, r, f, g):
    #nuB = barrier_func(X, Y, Z)*np.ones(r)
    #x = sympy.Symbol('x')
    #y = sympy.Symbol('y')
    #B = (x-1)**2 + (y-1)**2 - 0.25
    #i = 1
    #while i < r:    
    #    if i == 1:
    #        temp = np.zeros(12)
    #        temp[0] = 2*X
    #        temp[2] = 2*Y
    #        nuB[i] = np.dot(temp, f)
    #        #np.array([[B.diff(x)],[0],[B.diff(y)],[0],[0],[0],[0],[0],[0],[0],[0],[0]]).T # 1x12
    #        #nuB[0][0] = sympy.lambdify(x, nuB[0][0])
    #        #nuB[0][0] = nuB[0][0](X)
    #        #nuB[0][2] = sympy.lambdify(y, nuB[0][2])
    #        #nuB[0][2] = nuB[0][2](Y)
    #        #nuB[i] = np.sum(nuB[i])
    #        #nuB[i] = barrier_func(X, Y, Z)
    #    if i == 2:
    #        temp = np.zeros(12)
    #        temp[0] = 2
    #        temp[2] = 2
    #        nuB[i] = np.dot(temp, f)
    #        #np.array([[B.diff(x)],[0],[B.diff(y)],[0],[0],[0],[0],[0],[0],[0],[0],[0]]).T # 1x12
    #        # returns d(B)/dX for Lie Derivative calculation
    #        #nuB[i][0] = sympy.lambdify(x, nuB[i][0])
    #        #nuB[i][0] = nuB[i][0](X)
    #        #nuB[i][2] = sympy.lambdify(y, nuB[i][2])
    #        #nuB[i][2] = nuB[i][2](Y)
    #        #nuB[i] = np.sum(nuB[i])
    #
    #    elif i > 2:   
    #        nuB[i] = np.zeros(12)
    #    i += 1    
    a = barrier_func(X[0], X[2], X[4])
    b = 2*(X[0]*X[1] - X[1] + X[2]*X[3] - X[3])
    nuB = np.array([ a , b ])

    return nuB.T

def cbf_check(U, X):# U from (U = K * err_X) in LQR 
    ##   U_updated = optimise((Updated-U)^2) with constraint: 
    #               LieDerivative(barrier_func(), f) + LieDerivative(barrier_func(), G)*U_updated + alpha*barrier_func() >= 0
    # reference: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    M = np.eye(4)
    
    # nonlinear dynamics
    f, g = nonlinear_dynamics.f_g(X, U) 

    #print(f)
    #print(f.shape) #12x1

    #print(g)
    #print(g.shape) #4x12

    P = np.dot(M.T, M)
    q = -1*np.dot(M.T, U)
    
    B = barrier_func(X[0], X[2], X[4])
    LfB = lie_derivative(dBdx(X[0], X[2], X[4]), f)
    LgB = lie_derivative(dBdx(X[0], X[2], X[4]), g)
    alpha = 1#*np.ones(len(X)) # class kappa function constant #12x1 ## choose this as per the designing method in paper

    #print(LgB)
    #print(LgB.shape) #1x4

    #print(LfB)
    #print(LfB.shape) #1x1
    
    G = np.array([-1*LgB])
    h = np.array(([LfB + alpha*B]))

    #lie_gradB.T(1x12) * f(12x1)
    #lie_gradB.T(1x12) * g(12x4) => LgB (1x4) ; LgB(1x4) * u(4x1) => 1x1

    # LfB  + alpha*B >= -LgB*u
    # LfB, LgB will be scalars=> lie_gradB would be a vector

    #print(G)
    #print(G.shape) #1x4
    #print(type(G[0][0]))
    #print(h)
    #print(h.shape) #1x1
    #print(type(h))
    #print(G*U)
    #print(np.dot(G,U).shape) #1x1

    return quadprog_solve_qp(P, q, G, h) # optimisation

def ecbf_check(U, X, del_t, r = 2):# U from (U = K * err_X) in LQR; relative degree (r) as a hyper parameter  
    ## Relative degree != 1
    ##   U_updated = optimise((Updated-U)^2) with constraint: 
    #               LieDerivative(barrier_func(), f) + LieDerivative(barrier_func(), G)*U_updated + alpha*barrier_func() >= 0
    # reference: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    M = np.eye(4)  
    
    # nonlinear dynamics
    f, g = nonlinear_dynamics.f_g(X, U) #12x1, 12x4 

    P = np.dot(M.T, M)
    q = -1*np.dot(M.T, U)
    
    B = barrier_func(X[0], X[2], X[4]) # scalar
    nuB = nu_B(X, r, f, g)#rx12 or rx1?
    #LfB_r = lie_derivative(dBdX(X[0], X[2], X[4], r), f) #1x1
    # manually calculated:
    LfB_r = 2*( X[1]**3 + X[0]*X[1]*(X[1]/del_t) - X[0]*(X[1]/del_t) + X[3]**3 + X[2]*X[3]*(X[3]/del_t) - X[2]*(X[3]/del_t) )
    LfB_r_1 = lie_derivative(dBdX(X[0], X[2], X[4], r-1), f) #1x1
    LgB = lie_derivative(dBdX(X[0], X[2], X[4]), g) #4x1
    alpha = np.ones(r)# class kappa function row vector # 1xr

    LgLfB_r_1 = lie_derivative(lie_derivative(LfB_r_1, f).T, g)

    print(nuB)
    #print(dBdX(X[0], X[2], X[4]))
    #print(f)
    #print(g)

    #print(LgB)
    #print(LgB.shape) #1x4

    print(LfB_r)
    #print(LfB_r.shape) #1x1
    #print(LfB_r_1.shape) #1x1

    #print(LgLfB_r_1)

    # LfB^r + LgLfB^(r-1) U + alpha * nu_b >= 0
    G = np.array([-1*LgLfB_r_1])#LgB*LfB_r_1])
    h = np.array(([LfB_r + np.dot(alpha, nuB)]))
    
    print(G)
    #print(G.shape) #1x4
    #print(type(G[0][0]))
    #print(h)
    #print(h.shape) #1x1
    #print(type(h[0]))
    #print(G*U)
    #print(np.dot(G,U).shape) #1x1

    return quadprog_solve_qp(P, q, G.astype(float), h) # optimisation

def simple_check(U, X):
    print(barrier_func(X[0], X[2], X[4]))
    if (barrier_func(X[0], X[2], X[4])<0):
        U = np.array([m*g, 0, 0, 0])
    print(U)
    return U

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)

if __name__ == "__main__":
    # X = [x1, x2, y1, y2, z1, z2, phi1, phi2, theta1, theta2, psi1, psi2]

    relative_degree = 2
    U = np.array([1,1,1,1]).T
    X = np.array([0,0.1,0,0.1,0.1,0.1,1,1,1,1,0,1])
    
    print(ecbf_check(U, X))
    #print("**************")
    #print(ecbf_check(U, X, relative_degree))