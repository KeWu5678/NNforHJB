import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

# GLOBAL
## the grid
span = [0, 3]
grid = np.linspace(span[0], span[1], 1000)
guess = [1, 1]
## the boundary conditions
degree = 10  # interpolation degree of the control space
beta = 2  ## ODE parameter
tol = 0.01  # termination criteria
ini = [0, 0]


########################## TVBVP SLOVER ######################################
"""
Solve the TVBVP by shooting methods. The space of the control is interpolate by
the piecewise constant functions with value given by the array u.

input:  
    f:      functions; 
            The ODE
    u:      1xgrid.size array; 
            The control function
    span:   1x2 array; 
            the time span 
    ini:    1x2 array; 
            initial contition of ode
    guess:  1x2 array; 
            guess of the initial condition of the adjoint p
    tol:    1x2 array; 
            tolerance of the root-finding
    grid:   1d array 
            grid of the evaluation

return:   
    grad:   1 x grid.size array
            gradient evaluated on the grid
    
"""
def shooting(f, span, ini, u_coe, guess, grid, tol):
    
    ## Create the pieceeise constant callable u to pass it as argument in ODE
    def u(t):
        conditions = [(t >= grid[i]) & (t < grid[i+1]) for i in range(grid.size - 1)]
        return np.piecewise(t, conditions, u_coe)
   
    ##  Define the operator form guess of initial condition to residual of bc 
    ##  return: pb: 1x2 array;
    def residual(guess):
        guess_expand = np.zeros(4)
        guess_expand = [*ini, *guess]
        sol = solve_ivp(f, span, guess_expand, args=(u,), t_eval=grid)
        pb = sol.y[2:4, -1]
        return pb

    result = root(residual, guess, tol=tol)
    
    if not result.success:
        raise RuntimeError("Root finding failed: " + result.message)

    correct_ini = result.x
    # Solve the IVP with the correct slopes and evaluate on the fixed grid
    bvp_sol = solve_ivp(f, span, [*ini, *correct_ini], args=(u,), t_eval=grid)
    
    return bvp_sol
    


"""
Given the gradient and state in step k and k - 1, calculate the stepsize in k

input:  
    u_0:    1 x grid.size array
            Control function in time k -1
    u_1:    1 x grid.size array
            Control function in time k
    g_0:    1 x grid.size array
            Evaluation of gradient at time k -1 on the grid
    g_1:    1 x grid.size array
            Evaluation of gradient at time k on the grid
return:
    bb_1:   float
            BB-Step at odd k
    bb_2:   float
            BB-step at even k
"""
def BBstep(u_0, u_1, g_0, g_1, grid):
    bb1_nomi = 0
    bb1_denomi = 0
    bb2_nomi = 0
    bb2_denomi = 0
    # print('u0 = ', u_0)
    # print('u1 = ', u_1)
    # print('g0 = ', g_0)
    # print('g1 = ', g_1)
    for i in range(grid.size - 1):

        mesh = grid[i + 1] - grid[i]
        ## BB1
        bb1_nomi += (u_1[i] - u_0[i]) * (g_1[i] - g_0[i]) * mesh
        bb1_denomi += (u_1[i] - u_0[i]) ** 2 * mesh
        ## BB2
        bb2_nomi += (g_1[i] - g_0[i]) * 2 * mesh
        bb2_denomi += (u_1[i] - u_0[i]) * (g_1[i] - g_0[i]) * mesh
        # print(f"bb1_nomi: {bb1_nomi}, bb1_denomi: {bb1_denomi}")
        # print(f"bb2_nomi: {bb2_nomi}, bb2_denomi: {bb2_denomi}")
        # print("Differences in u:", u_1 - u_0)
        # print("Differences in g:", g_1 - g_0)

    bb_1 = bb1_nomi / bb1_denomi
    bb_2 = bb2_nomi / bb2_denomi

    return bb_1, bb_2



########################   THE ODE   ##########################################
"""
Define the Boudary value problem with the control parameter u
Input 
    t: dummy
    y: dummy
    u: function of u
"""
def BVP_VDP(t, y, u):
    y1, y2, p1, p2 = y  # Unpack the variables

    # Dynamics for y (states)
    dydt = np.zeros(4)  # Initialize the derivatives
    dydt[0] = y2
    dydt[1] = -y1 + y2 * (1 - y1**2) + u(t)

    # Dynamics for p (costates)
    dydt[2] = -p2 - 2 * y1 
    dydt[3] = 2 * p1 * y1 * y2 + p2 * y1**2 + p1 - p2 - 2 * y2

    return dydt




"""
Describe the gradient function of the system
Parameter:
    sol:    OdeResult
            Solution of the TVBVP
    u:      1 x grid.size
            Coefficient of the control
"""
def Grad_VDP(sol, u):
    # This piece of codes gives the gradient of the opt problem at u 
    grad = np.zeros(grid.size)
    p2 = np.zeros(grid.size)

    p2 = sol.y[3, :]  
    for i in range(grid.size - 1):  # Avoid out-of-bounds error
        grad[i] = p2[i] + 2 * beta * u[i]  # Compute value at x[i]

    return grad
    
    

##################  ERROR(NORM) ESTIMATION  ##################################
# Evaluate the L2^2 error of the gradient. With the left rectangle evaluation
"""
# Given the control and solution and TVBVP, evaluate the gradient.

# input:  
#     x:      1 x grid.size array
#             Control current
#     grid:   1 x grid.size array
#             linspace where x is evaluated

# return:   
#     norm:   float
#             The L2 norm of x.     
# """
def L2(x, grid):
    norm = 0  # Initialize norm

    for i in range(grid.size - 1):  # Avoid out-of-bounds error
        mesh = grid[i + 1] - grid[i]  # Mesh interval
        x_sq = x[i] ** 2 # Left evaluation
        norm += mesh * x_sq  # Compute measure for this interval
    return norm  # Return after the loop is complete



def Cost(y1, y2, u, grid):
    return L2(y1, grid) + L2(y2, grid) + beta * L2(u, grid)



##################### OPTIMIZATION ALGORITHM ##################################
# Define the initial control
u0 = np.zeros(grid.size)
alpha = 1 / 20
G0 = - alpha * Grad_VDP(shooting(BVP_VDP, span, ini, u0, guess, grid, tol), u0)

u1 = u0 - alpha * G0
sol = shooting(BVP_VDP, span, ini, u1, guess, grid, tol)
G1 =Grad_VDP(sol, u1)

# Ensure x0 is a numpy array
G = np.array(G0)
history_G = [G.copy()]
history_Sol = [sol.copy()]

k = 1
while L2(G1, grid) >= tol:
    s = u1 - u0
    y = G1 - G0
    # if k % 2 == 0:
    #     alpha = 1 / BBstep(u0, u1, G0, G1, grid)[0]
    # else:
    #     alpha = 1 / BBstep(u0, u1, G0, G1, grid)[1]
    print(" k = " , k)
    print(' alpha = ', alpha)
    print(" norm u =",  L2(u1,grid)) 
    print(" norm G =", L2(G1,grid))
    u0, G0 = u1, G1
    u1 = u1 - alpha * G1
    sol = shooting(BVP_VDP, span, ini, u1, guess, grid, tol)
    G1 =Grad_VDP(sol, u1)
    k += 1
    history_Sol.append(sol)
    history_G.append(G1)

sol = shooting(BVP_VDP, span, ini, u1, guess, grid, tol)
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="y1 (state)")
plt.plot(sol.t, sol.y[1], label="y2 (state)")
plt.plot(sol.t, sol.y[2], label="p1 (costate)")
plt.plot(sol.t, sol.y[3], label="p2 (costate)")



################### DISCARDED CODE ###########################################
    
    # x = bvp_sol.y[0, :]
    # y = bvp_sol.y[1, :]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(grid, x, y, label='Damped Oscillation', color='green')

    # # Add labels and title
    # ax.set_xlabel('t')
    # ax.set_ylabel('x(t)')
    # ax.set_zlabel('y(t)')
    # ax.set_title('3D Parametric Plot: t to x(t), y(t)')
    # ax.legend()

    # plt.show()
    















































