#implementing FTCS method in python
#I need to figure out if this is legit or not
import numpy as np
import matplotlib.pyplot as plt

#define gridsize and parameters
num_spaces = 100
grid = np.linspace(-1, 1, num_spaces+1)
epsilon = 0.05
delta = 0.1
dt = 0.001
dx = (2/(num_spaces))
v = 0.1

#central difference for first and second derivative
def deriv2(var):
    derivative = np.array([])
    for i in range(0, len(var)):
        if i == 0 or i == num_spaces:
            derivative = np.append(derivative, 101) 
        else:
            derivative = np.append(derivative, (var[i + 1] - 2 * var[i] + var[i-1])/(dx**2))
    return derivative

def deriv(var):
    #leave the first and last points empty for now
    derivative = np.array([])
    for i in range(1, len(var)-1):
        derivative = np.append(derivative, (var[i+1]-var[i-1])/(2*dx))
    return derivative

#initial conditions
c = 0 * grid + 1
rho = 0 * grid
phi = .25 * grid

fig, ax = plt.subplots()

for time in range(0, 1000):    
    #enforce boundary conditions
    #update phi based on poissons equation
    for i in range(0, num_spaces+1):
        if i == 0:
            newrow = np.concatenate([np.array([dx + delta * epsilon, -1 * delta * epsilon]), np.zeros(num_spaces-1)])
            matrix = newrow
        elif i == num_spaces:
            newrow = np.concatenate([np.zeros(num_spaces-1), np.array([-1 * delta * epsilon, dx + delta * epsilon])])
            matrix = np.vstack([matrix, newrow])
        else:
            newrow = np.concatenate([np.zeros(i-1), np.array([1, -2, 1]), np.zeros(num_spaces-i-1)])
            matrix = np.vstack([matrix, newrow])
            
    solution_vector = np.concatenate([[-v * dx], -1 * rho[1:num_spaces]/epsilon**2 * dx**2, [v * dx]])   
    phi = np.linalg.solve(matrix, solution_vector)

    deltaphi = phi[1] - phi[0]
    rho[0] = (c[1] * deltaphi + rho[1])/(1 - deltaphi**2)
    c[0] = -1 * (rho[1]-rho[0]) / deltaphi

    deltaphi = phi[num_spaces] - phi[num_spaces-1]
    c[num_spaces] = (c[num_spaces-1] - rho[num_spaces-1] * deltaphi)/(1 - deltaphi**2)
    rho[num_spaces] = (c[num_spaces-1] - c[num_spaces])/(deltaphi)

    #compute derivatives of everything
    dc = np.linspace(-1, 1, num_spaces+1)
    drho = np.linspace(-1, 1, num_spaces+1)
    dphi = np.linspace(-1, 1, num_spaces+1)

    drho[0] = (rho[1]-rho[0])/dx
    dc[0] = (c[1]-c[0])/dx
    dphi[0] = (phi[1]-phi[0])/dx

    drho[num_spaces] = (rho[num_spaces]-rho[num_spaces-1])/dx
    dc[num_spaces] = (c[num_spaces]-c[num_spaces-1])/dx
    dphi[num_spaces] = (phi[num_spaces]-phi[num_spaces-1])/dx

    drho[1:num_spaces] = deriv(rho)
    dc[1:num_spaces] = deriv(c)
    dphi[1:num_spaces] = deriv(phi)

    #compute dc/dt and drho/dt
    dcdt = epsilon * (deriv2(c) + drho * dphi + rho * deriv2(phi))
    dcdt[0] = epsilon*(1/dx)*(dc[1]+rho[1]*dphi[1])
    dcdt[num_spaces] = -1*epsilon*(1/dx)*(dc[num_spaces-1]+rho[num_spaces-1]*dphi[num_spaces-1])

    drhodt = epsilon * (deriv2(rho) + dc * dphi + c * deriv2(phi))
    drhodt[0] = epsilon*(1/dx)*(drho[1]+c[1]*dphi[1])
    drhodt[num_spaces] = -1*epsilon*(1/dx)*(drho[num_spaces-1]+c[num_spaces-1]*dphi[num_spaces-1])

    #update c and rho
    c += dt * dcdt
    rho += dt * drhodt


    if time % 200 == 0:
        #plt.plot(grid, rho, label="rho")
        ax.plot(grid, phi, label="phi @ 0."+str(time))
        #plt.plot(grid, c, label='C')

ax.set_xlabel("dimensionless length")
ax.set_ylabel("dimensionless potential")
plt.legend()
plt.show()