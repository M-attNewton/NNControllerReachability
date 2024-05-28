%% Reach-SDP with Forward Reachability
clear all; clc
close all;
addpath('./util');


%% System Parameters
% double integrator system (ts = 1.0)
A = [1 1; 0 1];
B = [0.5; 1];
sys.uub =  1;
sys.ulb = -1;
%sys.uub =  5;
%sys.ulb = 4;
load nnmpc_nets_di_1

weights_ = weights;
weights{1} = weights_{1}(1:5,:);
weights{2} = weights_{2}(1:5,1:5);
weights{3} = weights_{2}(6:10,1:5);
weights{4} = weights_{2}(1:5,6:10);
weights{5} = weights_{2}(6:10,6:10);
weights{6} = weights_{3}(1,1:5);

biases_ = biases;
biases{1} = biases_{1}(1:5);
biases{2} = biases_{1}(6:10);
biases{3} = biases_{1}(11:15);
biases{4} = biases_{2}(1:5);
biases{5} = biases_{2}(6:10);
biases{6} = biases_{3}(1);

C = eye(2);
n = size(B,1);
m = size(B,2);
sys.A = A;
sys.B = B;

dim_x = size(sys.A,1);
dim_u = size(sys.B,2);

%% get network parameters

dims(1) = size(weights{1},2);

num_layers = numel(weights)-1;

for i=1:num_layers
    dims(i+1) = size(weights{i},1);
end

dims(num_layers+2) = size(weights{end},1);

net = nnsequential(dims,'relu');
net.weights = weights;
net.biases = biases;

%% add projection layer
net_p = nnsequential([dims dims(end) dims(end)],'relu');

weights_p = weights;
weights_p{end+1} = -eye(dim_u);
weights_p{end+1} = -eye(dim_u);

biases_p = biases;
biases_p{end} = biases{end}-sys.ulb;
biases_p{end+1} =  sys.uub-sys.ulb;
biases_p{end+1} =  sys.uub;

net_p.weights = weights_p;
net_p.biases = biases_p;

%weights = weights_p;
%biases = biases_p;
%dims = [2,5,5,1,1];
AF = 'relu';

% Put in format for Julia
dims = net_p.dims;
weights = net_p.weights;
biases = net_p.biases;
for i = 1:(length(dims) - 1)
    weights2(1:dims(i+1), 1:dims(i), i) = weights{i};
    biases2(1:dims(i+1), i) = biases{i};
end

weights = weights2;
biases = biases2;

%filename = 'ReachSparsePsatz/netDoubleIntRandWeights2.mat';
%save(filename,'weights','biases','dims','AF')
%load netInvertedPendulum
load('inv_pend_MPC_bias_free.mat')
net_p.dims = [2,5,5,1];
net_p.weights{1} = W{1}; net_p.weights{2} = W{2}; net_p.weights{3} = W{3}; 
net_p.biases{1} = b{1}; net_p.biases{2} = b{2}; net_p.biases{3} = b{3}; 
net_p.activation = 'tanh';
% net_p.dims = net2.dims;
% net_p.weights = net2.weights;
% net_p.biases = net2.biases;
% net_p.activation = 'tanh';

%% Setup
% initial set
X0_b = [0.2; -0.1; 11; -10];% [0.2; -0.19; 0.2; -0.19] 
X0_poly = Polyhedron([1 0; -1 0; 0 1; 0 -1], X0_b);
X0 = X0_poly.outerApprox; % normalize the A matrix
X0_vec = X0;

%dx = 0.02; % shrink the tube for better visualization
%X0_poly_s = Polyhedron([1 0; -1 0; 0 1; 0 -1], X0_b-dx);
X0_poly_s = X0_poly; 

% reachability horizon
N = 6;

% facets of the output polytope
% A_out = [1 0; -1 0; 0 1; 0 -1; -1 1; 1 -1; 1 1; -1 -1];
% A_out = [1 0; -1 0; 0 1; 0 -1; -1 1; 1 -1; 1 1; -1 -1; 1 2; -1 -2; -1 2; 1 -2; 2 1; -2 -1; 2 -1; -2 1];
%A_out = [1 0; -1 0; 0 1; 0 -1; -1 1; 1 -1; 1 1; -1 -1; 1 2; -1 -2; 1 4; -1 -4; -1 2; 1 -2; -1 4; 1 -4];
%A_out = [1 0; -1 0; 0 1; 0 -1];

% ADD POLYTOPE CONSTUCTION
poly_size = 4;
A_out = zeros(poly_size,2);
 for i = 1:poly_size
        theta = (i-1)/poly_size*2*pi;
        A_out(i,:) = [cos(theta), sin(theta)];
 end
 

disp(['Starting FRS computation, N = ', num2str(N)]);


%% Gridding to Compute the Exact Reachable Sets
Xg_cell = {}; % grid-based reachable sets
Ug_cell = {}; % grid-based control sets
Xg = grid(X0_poly_s,40);
Xg_cell{end+1} = Xg;
for k = 1:N
    Xg_k = []; % one-step FRS at time k
    Ug_k = []; % one-step control set at time k
    for x = Xg_cell{end}'
        %u = fwd_prop(net,x);
        %x_next = A*x + B*proj(u(1),sys.ulb,sys.uub);
        %u = net_p.eval(x);
        %x_next = A*x + B*u;
        layer1 = tanh(W{1}*x);
        layer2 = tanh(W{2}*layer1);
        u = W{3}*layer2;
        delta_t = 0.01;
        mass = 0.15;
        grav = 10;
        leng = 0.5;
        mu = 0.5;
        sat_max = 1;
        x_next(1,1) = x(1) + delta_t*x(2);
        u = sign(u)*(min(abs(u), sat_max));
        q = x(1) - sin(x(1));
        x_next(2,1) = x(2) + delta_t*(-mass*grav*leng*q + mass*grav*leng*x(1) - mu*x(2) + u)/(mass*leng^2);
        Xg_k = [Xg_k; x_next'];
        Ug_k = [Ug_k; u(1)];
    end
    Xg_cell{end+1} = Xg_k;
    Ug_cell{end+1} = Ug_k;
end

%sys.A = [1, delta_t; -4*delta_t, (1-2*zeta*delta_t)];
%sys.B = [0; delta_t];

%% Reach-SDP
poly_cell = cell(1,N+1);
poly_cell{1,1}  = X0_vec;
poly_cell2 = poly_cell;

options.verbose = 0;
options.solver = 'mosek';
options.repeated = 0;

tssos_result = -[-0.3076282283968589
 -9.294572512038554
  0.2000000098316958
  8.419272907917518
 -0.4005739403778991
 -7.850778572665288
  0.2841927391228561
  7.068383439270248
 -0.4790817188753101
 -6.622705287572074
  0.35487657378812026
  5.917837684792651
 -0.5453087315589275
 -5.57914341987032
  0.4140549508870217
  4.938389475594892
 -0.6011000958015058
 -4.693589524942033
  0.4634388457549916
  4.104396917126717
 -0.6480359699869396
 -3.9443144846714655
  0.5044828149227912
  3.3926355686254372];

q = 1;
for i = 1:length(X0_vec)
    
    % polytopic initial set
    input_set  = X0_vec(i);
    poly_seq_vec   = X0_vec(i);
    poly_seq_vec2   = X0_vec(i);
    
    
    % forward reachability
    for k = 1:N
        net_p.activation = 'relu';
%        [b_out,~,~] =  reach_sdp(net_p,sys.A,sys.B,input_set.H,A_out',options);
        %[Y_min,Y_max,X_min,X_max,out_min,out_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net_p);
%         if k == 1
%         dim_x = dims(1);
%         dim_u = size(B,2);
% 
%         Fx = input_set.H(:,1:end-1);
%         fx = input_set.H(:,end);
% 
%         dim_px = length(fx);
% 
%         x = sdpvar(dim_x,1);
%         x_min = zeros(dim_x,1);
%         x_max = zeros(dim_x,1);
%         options = sdpsettings('solver','mosek','verbose',0);
%         for j=1:dim_x
%             optimize([Fx*x<=fx],x(j),options);
%             x_min(j,1) = value(x(j));
% 
%             optimize([Fx*x<=fx],-x(j),options);
%             x_max(j,1) = value(x(j));
%         end
%         else
%            x_min = min_temp;
%            x_max = max_temp;
%         end
% 
%         % Interval arithmetic to find the activation bounds
%         %[Y_min,Y_max] = net.interval_arithmetic(x_min,x_max);
% 
%         %X_min = net.activate(Y_min);
%         %X_max = net.activate(Y_max);
%         [Y_min,Y_max,X_min,X_max,out_min,out_max] = intervalBoundPropagation(x_min,x_max,dims(2:end-1),net_p);
%         A_temp = [1.0000     0
%     0.0000    1.0000
%    -1.0000    0.0000
%    -0.0000   -1.0000];
%         min_temp = A*x_min + B*out_min;
%         max_temp = A*x_max + B*out_max;
%         b_out(1) = max_temp(1);
%         b_out(2) = max_temp(2);
%         b_out(3) = -min_temp(1);
%         b_out(4) = -min_temp(2);
%         
        % shift horizon
%        input_set = Polyhedron(A_out, b_out);
        %tssos_set = Polyhedron(A_out, b_out);
        tssos_set = Polyhedron(A_out, tssos_result(q:q+(poly_size-1))); q = q + poly_size;
        
        % save results
%        poly_seq_vec = [poly_seq_vec Polyhedron(A_out, b_out)];
        poly_seq_vec2 = [poly_seq_vec2 tssos_set];
        
        % report
        disp(['Reach-SDP Progress: N = ', num2str(k), ', i = ',...
            num2str(i), ', volume: ', num2str(input_set.volume)]);
    end
%    poly_cell{1,i} = poly_seq_vec;
    poly_cell2{1,i} = poly_seq_vec2;
end


%% Plot results
figure('Renderer', 'painters')
hold on
% initial set
plot(X0_poly,'color','b','alpha',0.1)
% 
% % N-step FRS
% for i = 1:length(X0_vec)
%     for k = 1:N+1
%         FRS_V = poly_cell{1,i}(k).V;
%         FRS_V_bd = FRS_V(boundary(FRS_V(:,1), FRS_V(:,2), 0.0),:);
%         if k == 1
%             continue
% %             plot(FRS_V_bd(:,1),FRS_V_bd(:,2),'k-','LineWidth',2)
%         else
%             plot(FRS_V_bd(:,1),FRS_V_bd(:,2),'r-','LineWidth',5)
%         end
%     end
% end

% N-step FRS
for i = 1:length(X0_vec)
    for k = 1:N+1
        FRS_V2 = poly_cell2{1,i}(k).V;
        FRS_V_bd2 = FRS_V2(boundary(FRS_V2(:,1), FRS_V2(:,2), 0.0),:);
        if k == 1
            continue
%             plot(FRS_V_bd(:,1),FRS_V_bd(:,2),'k-','LineWidth',2)
        else
            plot(FRS_V_bd2(:,1),FRS_V_bd2(:,2),'k-','LineWidth',5)
        end
    end
end

% gridding states
for k = 2:N+1
    FRS = Xg_cell{k};
    FRS_bd = FRS(boundary(FRS(:,1), FRS(:,2), 0.5),:);
    plot(FRS_bd(:,1),FRS_bd(:,2),'b-','LineWidth',5)
end


grid on;
axis tight;
%xlim([-1,6]);
%ylim([-3,3]);
xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
set(gca,'LooseInset',get(gca,'TightInset'));
set(gcf,'position',[0,0,1000,1000])
ax = gca;
ax.FontSize = 22; 