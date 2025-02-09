using LinearAlgebra
using PyPlot
using Dates
using JSON

#setting up plot
fig, ax = subplots(3, 1, sharex=true, figsize=(5,7)) 

#GLOBAL VARIABLES
num_spaces = 200
dt = 10^-4
char_times = 1
num_time_steps = char_times / dt
grid = collect(LinRange(-1, 1, num_spaces+1))
dx = grid[2] - grid[1]
filename="no_rxn"

#PHYSICAL PARAMS
epsilon = 0.05
delta = 0.1
v = 0.5
D = 2.5678313e-10
L = 1e-5
kf = 0
kb = 0
Cb = 0.150
C_solid_Li = 77

parameters_dict = Dict(
    "num_spaces" => num_spaces, 
    "num__time_steps" => num_time_steps, 
    "dt" => dt, 
    "epsilon" => epsilon, 
    "delta" => delta, 
    "v" => v, 
    "kf" => kf, 
    "kb" => kb,
    "Cb" => Cb
)

#defining central difference for laplacian
function deriv2(var)   
    return (var[3:end] - 2 * var[2:end-1] + var[1:end-2]) / (dx^2)
end

#defining central difference for gradient
function deriv(var)
    return (var[3:end] - var[1:end-2]) / (2*dx)
end

#initial conditions: uniform concentration and charge neutrality
c = 0 .* grid .+ 1
rho = 0 .* grid
phi = v .* grid

#plot initial potential
ax[1].plot(grid, phi, label="initial phi")
ax[2].plot(grid, c+rho, label="initial +")
ax[3].plot(grid, c-rho, label="initial -")
Rleft = []

#storing gradients as global variable
dc = collect(LinRange(-1, 1, num_spaces+1))
drho = collect(LinRange(-1, 1, num_spaces+1))
dphi = collect(LinRange(-1, 1, num_spaces+1))

#defining poisson matrix
dl = push!(ones(num_spaces-1), -delta * epsilon)
du = pushfirst!(ones(num_spaces-1), -delta * epsilon)
d = push!(ones(num_spaces-1) .* -2, dx + delta * epsilon)
pushfirst!(d, dx + delta * epsilon)
matrix = LinearAlgebra.Tridiagonal(dl, d, du)

for time in 1:num_time_steps
    #enforce boundary conditions
    soln_vector = (-1 .* rho[2:end-1] ./ epsilon^2 .* dx^2)
    push!(soln_vector, v * dx)
    pushfirst!(soln_vector, -v * dx)

    phi = matrix \ soln_vector

    #BV flux condition at boundary 1:
    gradphiboundary = (phi[2] - phi[1])/dx
    phidrop = -v - phi[1]
    A = exp(-phidrop/2)
    B = exp(phidrop/2)

    BVmatrix = [(kf * A * dx + 1) (kf * A * dx - gradphiboundary*dx); (kf * A * dx - gradphiboundary*dx)  (kf * A * dx + 1)]
    BVsoln = [(c[2]+(kb/Cb)*B*dx); (rho[2]+(kb/Cb)*B*dx)]
    BVboundary = BVmatrix \ BVsoln
    c[1] = BVboundary[1]
    rho[1] = BVboundary[2]

    reaction_rate = kf * (c[1]+rho[1]) * A - kb/Cb * B
    push!(Rleft, reaction_rate)

    #BV flux condition at boundary 2
    gradphiboundary = (phi[num_spaces+1] - phi[num_spaces])/dx
    phidrop = v - phi[num_spaces+1]
    A = exp(-phidrop/2)
    B = exp(phidrop/2)

    BVmatrix = [(kf * A * dx + 1) (kf * A * dx + gradphiboundary*dx); (kf * A * dx + gradphiboundary*dx)  (kf * A * dx + 1)]
    BVsoln = [(c[num_spaces]+(kb/Cb)*B*dx); (rho[num_spaces]+(kb/Cb)*B*dx)]
    BVboundary = BVmatrix \ BVsoln
    c[num_spaces+1] = BVboundary[1]
    rho[num_spaces+1] = BVboundary[2]

    #compute derivatives of everything
    drho = deriv(rho)
    dc = deriv(c)
    dphi = deriv(phi)

    grad2c = deriv2(c)
    grad2rho = deriv2(rho)
    grad2phi = deriv2(phi)
    
    #update c and rho; boundary is taken care of at the top of the loop
    dcdt = epsilon .* (grad2c .+ drho .* dphi .+ rho[2:end-1] .* grad2phi)
    drhodt = epsilon .* (grad2rho .+ dc .* dphi .+ c[2:end-1] .* grad2phi)
    c[2:end-1] .+= (dt .* dcdt)
    rho[2:end-1] .+= (dt .* drhodt)

    if time in [num_time_steps/5 2*num_time_steps/5 3*num_time_steps/5 4*num_time_steps/5 num_time_steps] 
        ax[1].plot(grid, phi, label="phi @ t =" * string(time*dt))
        ax[2].plot(grid, c+rho, label="cation @ t =" * string(time*dt)) 
        ax[3].plot(grid, c-rho, label="anion @ t =" * string(time*dt))
    end
end

#plotting and labeling
ax[3].set_xlabel("Dimensionless position")    
ax[1].set_ylabel("potential")    
ax[2].set_ylabel("cation concentration")    
ax[3].set_ylabel("anion concentration")    
ax[1].legend(loc="center left", bbox_to_anchor=(0.9, 0.5))   
ax[2].legend(loc="center left", bbox_to_anchor=(0.9, 0.5))   
ax[3].legend(loc="center left", bbox_to_anchor=(0.9, 0.5))   
savefig("/Users/arun/Desktop/BVPlots/" * filename, dpi=300)

#dump parameters
open("/Users/arun/Desktop/BVPlots/" * filename * ".json", "w") do file
    JSON.print(file, parameters_dict)
end
