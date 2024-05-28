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
load netInvertedPendulum
net_p.dims = net2.dims;
net_p.weights = net2.weights;
net_p.biases = net2.biases;
net_p.activation = 'tanh';

%% Setup
% initial set
X0_b = [0.2; -0.19; 0.2; -0.19];
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
poly_size = 16;
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
        u = net_p.eval(x);
        %x_next = A*x + B*u;
        delta_t = 0.0025;
        mass = 0.15;
        grav = 10;
        leng = 0.5;
        mu = 0.5;
        sat_max = 0.7;
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

tssos_result = [0.20049999847977773
  0.25011205319550095
  0.26168747231926237
  0.23424075047848356
  0.17217641511103024
  0.0862074020447282
 -0.012911173841079303
 -0.11010970255301453
 -0.19047500004670231
 -0.2365357931778159
 -0.24665317986832613
 -0.21841719185446942
 -0.15639936851338979
 -0.06780422054344434
  0.031130778760574077
  0.12534797682203866
  0.20092380934388449
  0.24097064960419576
  0.24430225315822124
  0.21130478863839233
  0.14748114904732024
  0.06316053845720987
 -0.03073729296610528
 -0.11997145938960636
 -0.19087086339358078
 -0.22560423226903312
 -0.22599741353151684
 -0.1912820507769062
 -0.12658116861857197
 -0.04003533951122984
  0.052485783863618386
  0.13712081430809267
  0.2012852384498489
  0.23280810237839422
  0.22891406512846296
  0.19091422623745732
  0.12545195503085455
  0.0427007945164127
 -0.046603298057250796
 -0.1287260810853847
 -0.19119270937788224
 -0.21581612942694856
 -0.2075902607726327
 -0.16711016579162477
 -0.1000632548383586
 -0.015432687156483264
  0.07152189249255289
  0.14758841291305394
  0.20159126679095682
  0.22555107437175503
  0.2151847475852717
  0.17270998625950984
  0.10581383207809487
  0.024402138571434893
 -0.060652872623546294
 -0.1365209656666046
 -0.19144867003141652
 -0.2070804814722906
 -0.1911979228796582
 -0.14561015126574747
 -0.07646174382319498
  0.006501185124861066
  0.08840303531119409
  0.15687170783535984
  0.20184803921131642
  0.21907467526312288
  0.20298091107276636
  0.15658405866497085
  0.08840407480709686
  0.008192951980243243
 -0.07316776693204567
 -0.1434255433617181
 -0.19164603510137437
 -0.19929561341990093
 -0.17657735617146803
 -0.126467964658624
 -0.05540764792462524
  0.025956481566604777
  0.10336715971728355
  0.16509408580143925
  0.20206095283889583
  0.21330901627485585
  0.19212580396036744
  0.1421621931290601
  0.07289118657631605
 -0.006162450054618651
 -0.08427096204173463
 -0.14954066690470583
 -0.19179131025998844
 -0.19234270901725908
 -0.16355911800560433
 -0.10949691104264592
 -0.036788543534640984
  0.043214665014095925
  0.11666100259634334
  0.17237737566846914];

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