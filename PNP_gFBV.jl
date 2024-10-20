using LinearAlgebra
using PyPlot
using Dates
using JSON

#setting up plot
fig, ax = subplots(figsize=(7,7)) 

#defining parameters
num_spaces = 1000
num_time_steps = 10^6
dt = 1 / num_time_steps
grid = collect(LinRange(-1, 1, num_spaces+1))
dx = (2 / (num_spaces))
filename="no_reaction"

#physical parameters
epsilon = 0.05
delta = 0.1
v = 2
D = 2.5678313e-10
L = 1e-4
kf = 2.5 * 0
kb = 0.75 * 0
Cb = 0.150

parameters_dict = Dict(
    "num_spaces" => num_spaces, 
    "num_time_steps" => num_time_steps, 
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
    derivative = []
    for i in 1:length(var)
        if i == 1 || i == num_spaces+1
            push!(derivative, 0.0)
        else
            push!(derivative, (var[i + 1] - 2 * var[i] + var[i-1])/(dx^2))
        end
    end
    return derivative
end

#defining central difference for gradient
function deriv(var)
    derivative = []
    for i in 1:length(var)
        if i == 1
            push!(derivative, (var[2] - var[1])/dx)
        elseif i == num_spaces+1
            push!(derivative, (var[num_spaces+1] - var[num_spaces])/dx)
        else
            push!(derivative, (var[i+1]-var[i-1])/(2*dx))
        end
    end
    return derivative
end

#initial conditions: uniform concentration and charge neutrality
c = 0 .* grid .+ 1
rho = 0 .* grid
phi = v .* grid

#defining poisson solver matrix: includes robin boundary condition for potential
dl = push!(ones(num_spaces-1), -delta * epsilon)
du = pushfirst!(ones(num_spaces-1), -delta * epsilon)
d = push!(ones(num_spaces-1) .* -2, dx + delta * epsilon)
pushfirst!(d, dx + delta * epsilon)
matrix = LinearAlgebra.Tridiagonal(dl, d, du)

#plot initial potential
plot(grid, phi, label="initial")
Rleft = []

#storing gradients as global variable
dc = collect(LinRange(-1, 1, num_spaces+1))
drho = collect(LinRange(-1, 1, num_spaces+1))
dphi = collect(LinRange(-1, 1, num_spaces+1))

for time in 1:num_time_steps
#enforce boundary conditions
    #poissons equations for potential
    soln_vector = (-1 .* rho[2:(num_spaces)] ./ epsilon^2 .* dx^2)
    push!(soln_vector, v * dx)
    pushfirst!(soln_vector, -v * dx)

    phi = matrix \ soln_vector

    #BV flux condition at boundary 1:
    gradphiboundary = (phi[2] - phi[1])/dx
    phidrop = -v - phi[1]
    A = exp(-phidrop/2)
    B = exp(phidrop/2)

    BVmatrix = [(2 * kf * A * dx + 1) (2 * kf * A * dx - gradphiboundary*dx); (2 * kf * A * dx - gradphiboundary*dx)  (2 * kf * A * dx + 1)]
    BVsoln = [(c[2]+(kb/Cb)*B*dx); (rho[2]+(kb/Cb)*B*dx)]
    BVboundary = BVmatrix \ BVsoln
    c[1] = BVboundary[1]
    rho[1] = BVboundary[2]

    push!(Rleft, 2 * kf * (c[1]+rho[1]) * A - kb/Cb * B)

    #BV flux condition at boundary 2
    gradphiboundary = (phi[num_spaces+1] - phi[num_spaces])/dx
    phidrop = v - phi[num_spaces+1]
    A = exp(-phidrop/2)
    B = exp(phidrop/2)

    BVmatrix = [(2 * kf * A * dx + 1) (2 * kf * A * dx + gradphiboundary*dx); (2 * kf * A * dx + gradphiboundary*dx)  (2 * kf * A * dx + 1)]
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
    dcdt = epsilon .* (grad2c .+ drho .* dphi .+ rho .* grad2phi)
    drhodt = epsilon .* (grad2rho .+ dc .* dphi .+ c .* grad2phi)
    c[2:num_spaces] .+= (dt .* dcdt[2:num_spaces])
    rho[2:num_spaces] .+= (dt .* drhodt[2:num_spaces])

    if time in [num_time_steps/5 2*num_time_steps/5 3*num_time_steps/5 4*num_time_steps/5 num_time_steps] 
        #plt.plot(grid, rho, label="rho")
        plt.scatter(grid, phi, label="phi @ " * string(time*dt), s=0.5)
        #plt.plot(grid, c+rho, label="cation") 
    end
end
plt.legend()   
savefig("/Users/arun/Desktop/BVPlots/" * filename, dpi=300)

open("/Users/arun/Desktop/BVPlots/" * filename * ".json", "w") do file
    JSON.print(file, parameters_dict)
end
