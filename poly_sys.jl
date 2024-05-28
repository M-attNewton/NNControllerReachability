# General Neural Network Attempt 1
using TSSOS
using DynamicPolynomials
using Random
using MAT
using JuMP, Ipopt
using BenchmarkTools

#cd("C:\\Users\\lascat6145\\Documents\\ReachSparsePsatz")
include("NNPsatzChorFuncs.jl")

# Read NN from file
global net = matopen("netDuffingRandWeights6.mat")
#net = matopen("test.mat")
global biases = read(net,"biases")
global weights = read(net,"weights")
global dims = read(net,"dims")
global dims = round.(Int,dims)

global AF = "relu"

if AF == "sigmoid"
    # Mid point of sector
    global x_m = 1

    # Right side upper line L_ub
    global m1 = Model(Ipopt.Optimizer)
    @variable(m1,d1)
    @NLconstraint(m1, (1/(1+exp(-d1)))*(1 - 1/(1+exp(-d1))) == (1/(1+exp(-x_m)) - 1/(1+exp(-d1)))/(x_m - d1))
    @constraint(m1, d1 <= 0.9)
    optimize!(m1)
    global d1 = value(d1)
    global grad_L_ub = (1/(1+exp(-d1)))*(1 - 1/(1+exp(-d1)))
    global c_L_ub = (1/(1+exp(-d1))) - grad_L_ub*d1

    # Left side upper line L_lb
    global m2 = Model(Ipopt.Optimizer)
    @variable(m2,d2)
    @NLconstraint(m2, (1/(1+exp(-d2)))*(1 - 1/(1+exp(-d2))) == (1/(1+exp(x_m)) - 1/(1+exp(-d2)))/(-x_m - d2))
    @constraint(m2, d2 >= -0.9)
    optimize!(m2)
    global d2 = value(d2)
    global grad_L_lb = (1/(1+exp(-d2)))*(1 - 1/(1+exp(-d2)))
    global c_L_lb = (1/(1+exp(-d2))) - grad_L_lb*d2
    elseif AF == "tanh"
    # Mid point of sector
    global x_m = 1.1

    # Right side upper line L_ub
    global m1 = Model(Ipopt.Optimizer)
    @variable(m1,d1)
    @NLconstraint(m1, 1 - (tanh(d1))^2 == (tanh(x_m) - tanh(d1))/(x_m - d1))
    @constraint(m1, d1 <= x_m - 0.1)
    optimize!(m1)
    global d1 = value(d1)
    global grad_L_ub = 1 - (tanh(d1))^2
    global c_L_ub = tanh(d1) - grad_L_ub*d1

    # Left side upper line L_lb
    global m2 = Model(Ipopt.Optimizer)
    @variable(m2,d2)
    @NLconstraint(m2, 1 - (tanh(d2))^2 == (tanh(-x_m) - tanh(d2))/(-x_m - d2))
    @constraint(m2, d2 >= -x_m + 0.1)
    optimize!(m2)
    global d2 = value(d2)
    global grad_L_lb = 1 - (tanh(d2))^2
    global c_L_lb = tanh(d2) - grad_L_lb*d2
end

# Extract dimensions
global dim_in = dims[1]
#global dim_hidden = transpose(dims[2:end-1])
global dim_hidden = transpose(dims[2:end-1])
global dim_out = dims[end]

# Double integrator dynamics
#global Asys = [[1 1]; [0 1]]
#global Bsys = [0.5; 1]
#global Csys = [[1 0]; [0 1]]

# Number of edges of polytope, only when dim_out = 2
global dim_poly = 16

# Order of relaxation, can be 0,1,2,etc. or "min" that uses minimum for each clique
global order = 2#"min"

global opt_ts = 1
for ts in 1:6

# Create varaibles for optimisation
@polyvar x[1:sum(dim_hidden)]
@polyvar u[1:dim_in]
@polyvar d
global vars = [x;u;d]

global ineq_cons = 1
global eq_cons = 1

if ts == 1
# Input constraints
global u_min = [0.95; 0.95]
global u_max = [1.05; 1.05]

global ineq_cons = vcat(ineq_cons,u[1] - u_min[1])
global ineq_cons = vcat(ineq_cons,u_max[1] - u[1])
global ineq_cons = vcat(ineq_cons,u[2] - u_min[2])
global ineq_cons = vcat(ineq_cons,u_max[2] - u[2])

else
    for i in 1:dim_poly
        local theta = (i-1)/dim_poly*2*pi
        local temp = [cos(theta) sin(theta)]*u
        #global ineq_cons = vcat(ineq_cons, 1*(opt[i] - temp[1]))
        if dim_poly == 4
        if i == 1
            global u_max[1] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 2
            global u_max[2] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 3
            global u_min[1] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 4
            global u_min[2] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        end
    elseif dim_poly == 8
        if i == 1
            global u_max[1] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 2
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 3
            global u_max[2] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 4
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 5
            global u_min[1] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 6
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 7
            global u_min[2] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 8
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        end
    elseif dim_poly == 16
        if i == 1
            global u_max[1] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 2
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 3
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 4
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 5
            global u_max[2] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 6
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 7
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 8
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 9
            global u_min[1] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 10
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 11
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 12
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 13
            global u_min[2] = abs(opt[i])
            global ineq_cons = vcat(ineq_cons, 1*(-abs(opt[i]) - temp[1]))
        elseif i == 14
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 15
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        elseif i == 16
            global ineq_cons = vcat(ineq_cons, 1*(abs(opt[i]) - temp[1]))
        end
    end
    end
end

#elseif ts == 2

#global u_min = [2.006119289895729; 0.5000001640173773]
#global u_max = [3.9999987749283745; 1.5000154181251446]

#elseif ts == 3

#global u_min = [2.021984118593777; -0.4999995495575013]
#global u_max = [5.00001106060036; 0.500708618053097]

#elseif ts == 4

#global u_min = [1.3440564672445916; -1.4848351427984092]
#global u_max = [5.000716529926475; -0.46787540201369726]

#elseif ts == 5

#global u_min = [0.31790209895374383; -2.1153798796403636]
#global u_max = [4.03980702793107; -0.5603290644427259]

#elseif ts == 6

#global u_min = [-1.3017303632713981; -1.7201701579117754]
#global u_max = [2.996905512068644; -0.12751581361674874]

#end




#else
#    for i in 1:dim_poly
#        local theta = (i-1)/dim_poly*2*pi
#        local temp = [cos(theta) sin(theta)]*u
#        global ineq_cons = vcat(ineq_cons, 1*(opt[i] - temp[1]))
#        if i == 1
#            global u_max[1] = opt[i]
#        elseif i == 2
#            global u_max[2] = opt[i]
#        elseif i == 3
#            global u_min[1] = -opt[i]
#        elseif i == 4
#            global u_min[2] = -opt[i]
#        end
#    end
    #for j in 1:dim_poly
        #global ineq_cons = vcat(ineq_cons, -opt[j] - transpose(C[j,:])*u)
        #if opt[1] < opt[3]
    #        global u_min[1] = opt[1]
        #    global u_max[1] = opt[3]
    #    else
        #    global u_min[1] = opt[3]
    #        global u_max[1] = opt[1]
    #    end
    #    if opt[2] < opt[4]
    #        global u_min[2] = opt[2]
    #        global u_max[2] = opt[4]
    #    else
    #        global u_min[2] = opt[4]
    #        global u_max[2] = opt[2]
    #    end
        #global u_min[1] = opt[3]
        #global u_min[2] = opt[4]
        #global u_max[1] = -opt[1]
        #global u_max[2] = -opt[2]

    #    global ineq_cons = vcat(ineq_cons,u[1] - u_min[1])
    #    global ineq_cons = vcat(ineq_cons,u_max[1] - u[1])
    #    global ineq_cons = vcat(ineq_cons,u[2] - u_min[2])
    #    global ineq_cons = vcat(ineq_cons,u_max[2] - u[2])
    #end
#end

#global con_in1 = u[1] - u_min[1]
#global con_in2 = u_max[1] - u[1]
#global con_in3 = u[2] - u_min[2]
#global con_in4 = u_max[2] - u[2]

# Input constraints
#con_in1 = u - u_min
#con_in2 = u_max - u
#CHANGE TO POLYHEDRON

# Hidden layer constraints
#ineq_cons,eq_cons = hiddenLayerCons(u_min,u_max,u,x,dims,weights,biases)

#global ineq_cons = 1
#global eq_cons = 1

# IBP - get preprocessing values
local Y_min,Y_max,X_min,X_max,out_min,out_max = intBoundProp(u_min,u_max,dims,dim_hidden,weights,biases,AF)

if AF == "relu"
Ip = findall(Y_min -> Y_min >= 0.001,Y_min)
In = findall(Y_max -> Y_max <= -0.001,Y_max)
# don't actually need to compute Ipn I don't think
Ip2 = zeros(size(Ip,1))
for j in 1:size(Ip,1)
   Ip2[j] = round.(Int,Ip[j][1])
end
In2 = zeros(size(In,1))
for j in 1:size(In,1)
   In2[j] = round.(Int,In[j][1])
end

for j in 1:size(dim_hidden,2)
   if j == 1
       global x_curr_layer = x[1:dim_hidden[j]]
       global v = weights[1:dims[j+1], 1:dims[j],j]*u + biases[1:dims[j+1],j]
   else
       global x_prev_layer = x[sum(dim_hidden[1:j-2]) + 1 : sum(dim_hidden[1:j-1])]
       global x_curr_layer = x[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
       global v = weights[1:dims[j+1], 1:dims[j],j]*x_prev_layer + biases[1:dims[j+1],j]
   end
   for k in 1:dim_hidden[j]
       local node_num = sum(dim_hidden[1:j-1]) + k
       if in(node_num).(Ip) == 1
           global ineq_cons = vcat(ineq_cons, x_curr_layer[k])
           global eq_cons = vcat(eq_cons, x_curr_layer[k] - v[k])
           global eq_cons = vcat(eq_cons, x_curr_layer[k]*(x_curr_layer[k] - v[k]))
       elseif in(node_num).(In) == 1
           global eq_cons = vcat(eq_cons, x_curr_layer[k])
           global ineq_cons = vcat(ineq_cons, x_curr_layer[k] - v[k])
           global eq_cons = vcat(eq_cons, x_curr_layer[k]*(x_curr_layer[k] - v[k]))
       else
           global ineq_cons = vcat(ineq_cons, x_curr_layer[k])
           global ineq_cons = vcat(ineq_cons, x_curr_layer[k] - v[k])
           global eq_cons = vcat(eq_cons, x_curr_layer[k]*(x_curr_layer[k] - v[k]))
       end
       global ineq_cons = vcat(ineq_cons, -(x_curr_layer[k] - X_min[sum(dim_hidden[1:j-1]) + k])*(x_curr_layer[k] - X_max[sum(dim_hidden[1:j-1]) + k]))
   end
end
elseif AF == "sigmoid"

#    # Mid point of sector
#    x_m = 1
#
#    # Right side upper line L_ub
#    m1 = Model(Ipopt.Optimizer)
#    @variable(m1,d1)
#    @NLconstraint(m1, (1/(1+exp(-d1)))*(1 - 1/(1+exp(-d1))) == (1/(1+exp(-x_m)) - 1/(1+exp(-d1)))/(x_m - d1))
#    @constraint(m1, d1 <= 0.9)
#    optimize!(m1)
#    d1 = value(d1)
#    grad_L_ub = (1/(1+exp(-d1)))*(1 - 1/(1+exp(-d1)))
#    c_L_ub = (1/(1+exp(-d1))) - grad_L_ub*d1
#
#    # Left side upper line L_lb
#    m2 = Model(Ipopt.Optimizer)
#    @variable(m2,d2)
#    @NLconstraint(m2, (1/(1+exp(-d2)))*(1 - 1/(1+exp(-d2))) == (1/(1+exp(x_m)) - 1/(1+exp(-d2)))/(-x_m - d2))
#    @constraint(m2, d2 >= -0.9)
#    optimize!(m2)
#    d2 = value(d2)
#    grad_L_lb = (1/(1+exp(-d2)))*(1 - 1/(1+exp(-d2)))
#    c_L_lb = (1/(1+exp(-d2))) - grad_L_lb*d2

    for j in 1:size(dim_hidden,2)
       if j == 1
           local x_curr_layer = x[1:dim_hidden[j]]
           local v = weights[1:dims[j+1], 1:dims[j],j]*u + biases[1:dims[j+1],j]
           local X_min_curr_layer = X_min[1:dim_hidden[j]]
           local X_max_curr_layer = X_max[1:dim_hidden[j]]
           local Y_min_curr_layer = Y_min[1:dim_hidden[j]]
           local Y_max_curr_layer = Y_max[1:dim_hidden[j]]
       else
           local x_prev_layer = x[sum(dim_hidden[1:j-2]) + 1 : sum(dim_hidden[1:j-1])]
           local x_curr_layer = x[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local v = weights[1:dims[j+1], 1:dims[j],j]*x_prev_layer + biases[1:dims[j+1],j]
           local X_min_curr_layer = X_min[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local X_max_curr_layer = X_max[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local Y_min_curr_layer = Y_min[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local Y_max_curr_layer = Y_max[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
       end
       for k in 1:dim_hidden[j]
           # Two sector constraints
           if Y_max_curr_layer[k] > 0 && Y_min_curr_layer[k] < 0
            # Sector in right hand plane
            if Y_max_curr_layer[k] > x_m
                local grad1a = (X_max_curr_layer[k] - (1/(1+exp(-x_m))))/(Y_max_curr_layer[k] - x_m)
                local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k];
                # Check for overlapping sectors
                if X_min_curr_layer[k] >  Y_min_curr_layer[k]*grad1a + c1a
                    local grad1a = (X_min_curr_layer[k] - (1/(1+exp(-x_m))))/(Y_min_curr_layer[k] - x_m)
                    local c1a = X_min_curr_layer[k] - grad1a*Y_min_curr_layer[k]
                end
            else
                local grad1a = 0;
                #c1a = X_max_curr_layer(k);
                local c1a = (1/(1+exp(-x_m)))
            end
            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad1a*v[k] + c1a))*((grad_L_ub*v[k] + c_L_ub) - x_curr_layer[k]) )

            # Sector in left hand plane
            if Y_min_curr_layer[k] < -x_m
                local grad2a = (X_min_curr_layer[k] - (1/(1+exp(x_m))) )/(Y_min_curr_layer[k] - -x_m)
                local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k];
                # Check for overlapping sectors
                if X_max_curr_layer[k] <  Y_max_curr_layer[k]*grad2a + c2a
                    local grad2a = (X_max_curr_layer[k] - (1/(1+exp(x_m))) )/(Y_max_curr_layer[k] - -x_m)
                    local c2a = X_max_curr_layer[k] - grad2a*Y_max_curr_layer[k]
                end
            else
                local grad2a = 0;
                #c2a = X_min_curr_layer(k);
                local c2a = (1/(1+exp(x_m)))
            end
            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad2a*v[k] + c2a))*((grad_L_lb*v[k] + c_L_lb) - x_curr_layer[k]) )

        elseif Y_max_curr_layer[k] < 0 && Y_min_curr_layer[k] < 0
            local Ysec = (Y_max_curr_layer[k] + Y_min_curr_layer[k])/2
            local Xsec = (1/(1+exp(-Ysec)))

            local grad1a = (X_max_curr_layer[k] - Xsec)/(Y_max_curr_layer[k] - Ysec)
            local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k]

            local grad2a = (X_min_curr_layer[k] - Xsec)/(Y_min_curr_layer[k] - Ysec)
            local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k]

            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad2a*v[k] + c2a))*((grad1a*v[k] + c1a) - x_curr_layer[k]) )

        elseif Y_max_curr_layer[k] > 0 && Y_min_curr_layer[k] > 0
            local Ysec = (Y_max_curr_layer[k] + Y_min_curr_layer[k])/2
            local Xsec = (1/(1+exp(-Ysec)))

            local grad1a = (X_max_curr_layer[k] - Xsec)/(Y_max_curr_layer[k] - Ysec)
            local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k]

            local grad2a = (X_min_curr_layer[k] - Xsec)/(Y_min_curr_layer[k] - Ysec)
            local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k]

            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad1a*v[k] + c1a))*((grad2a*v[k] + c2a) - x_curr_layer[k]) )

        end
        global ineq_cons = vcat(ineq_cons, -(x_curr_layer[k] - X_min[sum(dim_hidden[1:j-1]) + k])*(x_curr_layer[k] - X_max[sum(dim_hidden[1:j-1]) + k]))
        end
    end
elseif AF == "tanh"

    # Mid point of sector
    #x_m = 1.1
#    x_m = 1.1

    # Right side upper line L_ub
#    m1 = Model(Ipopt.Optimizer)
#    @variable(m1,d1)
#    @NLconstraint(m1, 1 - (tanh(d1))^2 == (tanh(x_m) - tanh(d1))/(x_m - d1))
#    @constraint(m1, d1 <= x_m - 0.1)
#    optimize!(m1)
#    d1 = value(d1)
#    grad_L_ub = 1 - (tanh(d1))^2
#    c_L_ub = tanh(d1) - grad_L_ub*d1
#
    # Left side upper line L_lb
#    m2 = Model(Ipopt.Optimizer)
#    @variable(m2,d2)
#    @NLconstraint(m2, 1 - (tanh(d2))^2 == (tanh(-x_m) - tanh(d2))/(-x_m - d2))
#    @constraint(m2, d2 >= -x_m + 0.1)
#    optimize!(m2)
#    d2 = value(d2)
#    grad_L_lb = 1 - (tanh(d2))^2
#    c_L_lb = tanh(d2) - grad_L_lb*d2

    for j in 1:size(dim_hidden,2)
       if j == 1
           local x_curr_layer = x[1:dim_hidden[j]]
           local v = weights[1:dims[j+1], 1:dims[j],j]*u + biases[1:dims[j+1],j]
           local X_min_curr_layer = X_min[1:dim_hidden[j]]
           local X_max_curr_layer = X_max[1:dim_hidden[j]]
           local Y_min_curr_layer = Y_min[1:dim_hidden[j]]
           local Y_max_curr_layer = Y_max[1:dim_hidden[j]]
       else
           local x_prev_layer = x[sum(dim_hidden[1:j-2]) + 1 : sum(dim_hidden[1:j-1])]
           local x_curr_layer = x[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local v = weights[1:dims[j+1], 1:dims[j],j]*x_prev_layer + biases[1:dims[j+1],j]
           local X_min_curr_layer = X_min[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local X_max_curr_layer = X_max[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local Y_min_curr_layer = Y_min[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local Y_max_curr_layer = Y_max[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
       end
       for k in 1:dim_hidden[j]
           # Two sector constraints
           if Y_max_curr_layer[k] > 0 && Y_min_curr_layer[k] < 0
            # Sector in right hand plane
            if Y_max_curr_layer[k] > x_m
                local grad1a = (X_max_curr_layer[k] - tanh(x_m))/(Y_max_curr_layer[k] - x_m)
                local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k];
                # Check for overlapping sectors
                if X_min_curr_layer[k] >  Y_min_curr_layer[k]*grad1a + c1a
                    local grad1a = (X_min_curr_layer[k] - tanh(x_m))/(Y_min_curr_layer[k] - x_m)
                    local c1a = X_min_curr_layer[k] - grad1a*Y_min_curr_layer[k]
                end
            else
                local grad1a = 0;
                #c1a = X_max_curr_layer(k);
                local c1a = tanh(x_m)
            end
            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad1a*v[k] + c1a))*((grad_L_ub*v[k] + c_L_ub) - x_curr_layer[k]) )

            # Sector in left hand plane
            if Y_min_curr_layer[k] < -x_m
                local grad2a = (X_min_curr_layer[k] - tanh(-x_m) )/(Y_min_curr_layer[k] - -x_m)
                local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k];
                # Check for overlapping sectors
                if X_max_curr_layer[k] <  Y_max_curr_layer[k]*grad2a + c2a
                    local grad2a = (X_max_curr_layer[k] -  tanh(-x_m) )/(Y_max_curr_layer[k] - -x_m)
                    local c2a = X_max_curr_layer[k] - grad2a*Y_max_curr_layer[k]
                end
            else
                local grad2a = 0;
                #c2a = X_min_curr_layer(k);
                local c2a =  tanh(-x_m)
            end
            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad2a*v[k] + c2a))*((grad_L_lb*v[k] + c_L_lb) - x_curr_layer[k]) )

        elseif Y_max_curr_layer[k] < 0 && Y_min_curr_layer[k] < 0
            local Ysec = (Y_max_curr_layer[k] + Y_min_curr_layer[k])/2
            local Xsec = tanh(Ysec)

            local grad1a = (X_max_curr_layer[k] - Xsec)/(Y_max_curr_layer[k] - Ysec)
            local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k]

            local grad2a = (X_min_curr_layer[k] - Xsec)/(Y_min_curr_layer[k] - Ysec)
            local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k]

            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad2a*v[k] + c2a))*((grad1a*v[k] + c1a) - x_curr_layer[k]) )

        elseif Y_max_curr_layer[k] > 0 && Y_min_curr_layer[k] > 0
            local Ysec = (Y_max_curr_layer[k] + Y_min_curr_layer[k])/2
            local Xsec = tanh(Ysec)

            local grad1a = (X_max_curr_layer[k] - Xsec)/(Y_max_curr_layer[k] - Ysec)
            local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k]

            local grad2a = (X_min_curr_layer[k] - Xsec)/(Y_min_curr_layer[k] - Ysec)
            local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k]

            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad1a*v[k] + c1a))*((grad2a*v[k] + c2a) - x_curr_layer[k]) )

        end
        global ineq_cons = vcat(ineq_cons, -(x_curr_layer[k] - X_min[sum(dim_hidden[1:j-1]) + k])*(x_curr_layer[k] - X_max[sum(dim_hidden[1:j-1]) + k]))
        end
    end
end

global ineq_cons = ineq_cons[2:end]
if size(eq_cons,1) >= 2
global eq_cons = eq_cons[2:end]
end

# Output constraints
global v_out = weights[1:dims[end],1:dims[end-1],end]*x[end - dim_hidden[end] + 1 : end] + biases[1:dims[end],end]
global delta_t = 0.1
global zeta = 0.3
global x_tplus1 = u[1] + delta_t*u[2]
global x_tplus1 = vcat(x_tplus1, (1 - 2*delta_t*zeta)*u[2] - delta_t*u[1] - delta_t*(u[1])^3 + delta_t*v_out)
#global x_tplus1 = Asys*u + Bsys.*v_out

#global x_tplus1 = Asys*u + Bsys.*x[end]

#x_tplus1 = Bsys.*v_out

# Controller saturation
#con_out1 = v_out[1] + 1.0
#con_out2 = -v_out[1] + 1.0

# CHANGE OUTPUT CONSTRAINTS
global opt = zeros(dim_poly,1)
global C = zeros(dim_poly,2)
for i in 1:dim_poly
    local theta = (i-1)/dim_poly*2*pi
    global C[i,:] = [cos(theta) sin(theta)]
    local f = -d
    local g0 = [cos(theta) sin(theta)]*x_tplus1 - d
    #local f = d
    #local g0 = d - [cos(theta) sin(theta)]*x_tplus1
    local pop = vcat(f, g0)
    #local pop = vcat(pop,con_in1)
    #local pop = vcat(pop,con_in2)
    #local pop = vcat(pop,con_in3)
    #local pop = vcat(pop,con_in4)
    #local pop = vcat(pop,con_out1)
    #local pop = vcat(pop,con_out2)
    local pop = vcat(pop, ineq_cons)
    local pop = vcat(pop, eq_cons)
    if size(eq_cons,1) >= 2
        global opt[i],sol,data = cs_tssos_first(pop, vars, order, numeq=size(eq_cons,1), TS="block", solution=true)
    else
        global opt[i],sol,data = cs_tssos_first(pop, vars, order, numeq=0, TS="block", solution=true)
        #@btime cs_tssos_first(pop, vars, 2, numeq=0, TS="block", solution=true)
    end
end
#global opt = -opt
global opt_ts = vcat(opt_ts,opt)
end
global opt_ts = opt_ts[2:end]
