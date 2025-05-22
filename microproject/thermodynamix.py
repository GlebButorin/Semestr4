from dolfin import *
from msh2xdmf import import_mesh, msh2xdmf
import numpy as np


# Constants for air (diatomic gas)
R = 8.31           # Universal gas constant
c_v = 5/2 * R       # Specific heat at constant volume (J/kg·K)
c_p = c_v + R       # Specific heat at constant pressure (J/kg·K)
gamma = c_p / c_v   # Heat capacity ratio (≈1.4 for air)
#mu_air = 1.48e-5
mu_air = 0    # Dynamic viscosity (kg/m·s)
#k_air = 0.0257
k_air = 0      # Thermal conductivity (W/m·K)
g_value = 9.81  # Gravity (m/s²)
p_0 = 100000.
T_ambient = 300.
r_steam = 0.0211 #saturated steam density at T = 300 K
mu = 0.029
#rho = 1.23
Gamma_d = 0.0098




def simulate_mountain_flow_with_time():
    # Load mesh (tags: "inlet", "outlet", "mountain", "ground", "top")
    mesh, boundaries, association_table = import_mesh(prefix="Mountain", directory="microproject/Mountain/build", dim=3)
    
    # Mixed function space (velocity P2, pressure P1, temperature P1)
    P2 = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    element = MixedElement([P2, P1, P1, P1])  # [Velocity, Pressure, Temperature, Humidity]
    W = FunctionSpace(mesh, element)
    
    # Time parameters
    total_time = 300
      # Total simulation time (s)
    num_steps = 1500   # Number of time steps
    dt = total_time / num_steps  # Time step size
    save_every = 5
    
    # Test and trial functions
    (v, r, s, phi) = TestFunctions(W)
    upTq = Function(W)      # Current solution
    upTq_n = Function(W)    # Solution at previous time step
    
    # Split functions
    
    u, p, T, q = split(upTq)
    u_n, p_n, T_n, q_n = split(upTq_n)


    dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})


    # Gravity and Boussinesq
    g = Constant((0.0, 0.0, -g_value))
    rho = (p_0*mu/(R*T_ambient)) * (1 + (p-p_0)/p_0 - (T-T_ambient)/T_ambient)
    rho_n = (p_0*mu/(R*T_ambient)) * (1 + (p_n-p_0)/p_0 - (T_n-T_ambient)/T_ambient)


    
    # Time-stepping (Crank-Nicolson)
    theta = Constant(1)  # Implicit/Explicit blending
    
    # First Law of Thermodynamics (internal energy)

    
    h = 1/mu * c_p*T  # Enthalpy for diatomic gas
    h_n = 1/mu * c_p*T_n



    F_energy = (
        rho*(h - h_n)/dt*s*dx +
        div(rho*h*u)*s*dx +
        (1-theta)*div(rho_n*h_n*u_n)*s*dx  -
        theta * rho * inner(g, u) * s * dx -          
        (1 - theta) * rho_n * inner(g, u_n) * s * dx  
    )

    

    # Saturation specific humidity 
    

    L_v = 2.5e6  # J/kg
    
    # Humidity transport equation
    F_humidity = (
        (q - q_n)/dt*phi*dx + 
        div(q*u)*phi*dx
    )
    
    # Momentum equation 
    
    F_momentum = (
        rho * inner((u - u_n)/dt, v) * dx +                     # Acceleration
        theta * rho * inner(dot(grad(u), u), v) * dx +           # Convection (current)
        (1 - theta) * rho_n * inner(dot(grad(u_n), u_n), v) * dx - 
        theta * p * div(v) * dx -                                # Pressure (current)
        (1 - theta) * p_n * div(v) * dx -
        theta * rho * inner(g, v) * dx -                         # Gravity (current)
        (1 - theta) * rho_n * inner(g, v) * dx 
    )
    
    
    # Continuity equation (mass conservation)
    F_continuity = (
    (rho - rho_n) / dt * r * dx +  # Time derivative of density
    div(rho * u) * r * dx)           # Divergence of mass flux

    


    
    z0 = 100  # Roughness height (m)
    u_star = 10  # Friction velocity (m/s)


    
    # Combined weak form
    F = F_momentum + F_continuity  + F_energy + F_humidity

    
    # 1. Inlet: Logarithmic wind profile + ambient temperature + humidity

    inflow_profile = Expression(
        ('u_star * (std::log((x[2] + z0)/z0))', '0', '0'), 
        u_star=u_star, z0=z0, degree=2
    )


    str_p_outlet = "-p0 / T0 * Gamma_d * x[2] + p0 * R / (g * mu * T0) * Gamma_d + " \
    "(p0 - p0 * R / (g * mu * T0) * Gamma_d) * pow(e, - (g * mu) / (R * T0) * x[2])"

    # Inlet humidity profile (typically decreases with height)
    q_inlet = Expression("q0*exp(-x[2]/H_q)", q0=0.8 * r_steam, H_q=800, degree=2)
    bcq_inlet = DirichletBC(W.sub(3), q_inlet, boundaries, association_table["inlet"])




    bcu_inlet = DirichletBC(W.sub(0), inflow_profile, boundaries, association_table["inlet"])

    str_T_inlet = "T0 - Gamma_d * x[2]"
    T_inlet = Expression(str_T_inlet, 
                    T0=T_ambient, Gamma_d=Gamma_d, degree=2)
    bcT_inlet = DirichletBC(W.sub(2), T_inlet, boundaries, association_table["inlet"])

    
    # 2. Mountain: No-slip + adiabatic


    bcu_mountain = DirichletBC(W.sub(0), Constant((0, 0, 0)), boundaries, association_table["mountain"])

    

    # 3. Outlet: stress-free velocity + ambient temperature

    bcu_outlet = DirichletBC(W.sub(0), inflow_profile, boundaries, association_table["outlet"])
    bcq_outlet = DirichletBC(W.sub(3), q_inlet, boundaries, association_table["outlet"])
    
    # 4. Ground: No-slip (optional logarithmic profile)
    bcu_ground = DirichletBC(W.sub(0), Constant((0, 0, 0)), boundaries, association_table["ground"])

    
    # 5. Top/Sides: Slip (u_z = 0) + adiabatic
    bc_slip_1 = DirichletBC(W.sub(0).sub(2), Constant(0), boundaries, association_table["top"])  # No vertical flow
    bc_slip_2 = DirichletBC(W.sub(0).sub(1), Constant(0), boundaries, association_table["sides"])  # No horizontal flow

    
    # Collect all BCs
    bcs = [bcu_inlet, bcu_mountain, bcu_ground, bc_slip_1, bc_slip_2, 
           bcT_inlet, bcu_outlet, bcq_inlet, bcq_outlet]

    # Set initial conditions (ambient pressure and temperature, zero velocity)
    
    
    upTq_n.assign(project(Expression(('u_star * (std::log((x[2] + z0)/z0))', '0', '0', str_p_outlet, str_T_inlet, "q0*exp(-x[2]/H_q)"), 
                                        u_star = u_star, z0 = z0, p0=p_0, T0=T_ambient, Gamma_d=Gamma_d, g=g_value, mu=mu, R=R, e = 2.7182818, 
                                        q0=0.8 * r_steam, H_q=800, degree=2), W))

    
    velocity_pvd = File("results/velocity.pvd")
    pressure_pvd = File("results/pressure.pvd")
    temperature_pvd = File("results/temperature.pvd")
    humidity_pvd = File("results/humidity.pvd")

    # 1. Create FunctionSpace for relative humidity
    Q = FunctionSpace(mesh, "CG", 1)  # Same order as your humidity field
    q_rel_output = Function(Q, name="RelativeHumidity")


 # Time-stepping loop
    t = 0.0
    for n in range(num_steps):
        t += dt
        print(f"Time step {n+1}/{num_steps}, t = {t:.2f} s")
        
        # Solve the system
        solve(F == 0, upTq, bcs, solver_parameters = {
    "nonlinear_solver": "newton",
    "newton_solver": {
        "linear_solver": "gmres",
        "preconditioner": "hypre_euclid",
        "maximum_iterations": 100,
        "relative_tolerance": 1e-22,
        "absolute_tolerance": 1e+1,
        #"relaxation_parameter": 1,  #  under-relaxation
        "error_on_nonconvergence": False,
        "report": True,
        "krylov_solver": {
            "nonzero_initial_guess": True,
            "monitor_convergence": True,
            "maximum_iterations": 100,
            "relative_tolerance": 1e-2
        }
    }
})
        
        # Save results at selected timestamps
        u_sol, p_sol, T_sol, q_sol= upTq.split()


        # 2. Compute q_sat as a Function
        q_sat = project(0.018/(8.31 * T_sol) * 1e5 * exp((2.5e6*0.018/8.31*(-1/T_sol + 1/373))), Q)

        # 3. Compute and project relative humidity
        q_rel = project(q_sol/q_sat, Q)

        # 4. Save to file
        q_rel_output.assign(q_rel)
        

        if (n % save_every == 0):
            velocity_pvd << (u_sol, t)
            pressure_pvd << (p_sol, t)
            temperature_pvd << (T_sol, t)
            humidity_pvd << (q_rel_output, t)

        
        # Update previous solution
        upTq_n.assign(upTq)
    
    return upTq

# Run simulation

upTq = simulate_mountain_flow_with_time()
