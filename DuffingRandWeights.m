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
load netDuffingRandWeights6
net_p.dims = net2.dims;
net_p.weights = net2.weights;
net_p.biases = net2.biases;
net_p.activation = 'relu';

%% Setup
% initial set
X0_b = [1.05; -0.95; 1.05; -0.95];
X0_poly = Polyhedron([1 0; -1 0; 0 1; 0 -1], X0_b);
X0 = X0_poly.outerApprox; % normalize the A matrix
X0_vec = X0;

%dx = 0.02; % shrink the tube for better visualization
dx = 0.0;
X0_poly_s = Polyhedron([1 0; -1 0; 0 1; 0 -1], X0_b-dx);

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
        delta_t = 0.1;
        zeta = 0.3;
        x_next(1,1) = x(1) + delta_t*x(2);
        x_next(2,1) = (1 - 2*zeta*delta_t)*x(2) - delta_t*x(1) - delta_t*x(1)^3 + delta_t*u;
        Xg_k = [Xg_k; x_next'];
        Ug_k = [Ug_k; u(1)];
    end
    Xg_cell{end+1} = Xg_k;
    Ug_cell{end+1} = Ug_k;
end

sys.A = [1, delta_t; -4*delta_t, (1-2*zeta*delta_t)];
sys.B = [0; delta_t];

%% Reach-SDP
poly_cell = cell(1,N+1);
poly_cell{1,1}  = X0_vec;
poly_cell2 = poly_cell;

options.verbose = 0;
options.solver = 'mosek';
options.repeated = 0;

tssos_result = [1.154999599252645
  1.4368436490940975
  1.499941096271236
  1.3348877345719325
  1.0062619529140802
  0.5259340349363875
 -0.03446273727423283
 -0.589613092301746
 -1.045000940673046
 -1.314562620647556
 -1.383994296053038
 -1.2427255727600197
 -0.8722376854373034
 -0.3676702789699119
  0.19287201775877702
  0.7240507825428043
  1.2516231327842613
  1.477298280012998
  1.4782106921943432
  1.2949440132595935
  0.9229619400744348
  0.41046657527174363
 -0.1645187451279065
 -0.710025290564173
 -1.1362282994060633
 -1.370774510161533
 -1.3966336475029488
 -1.1693500615706414
 -0.7552919238322873
 -0.2262471414197421
  0.3372417201912527
  0.8540736641694723
  1.3354908715570766
  1.4891845862207298
  1.4473499032652086
  1.214528208724922
  0.797689665851145
  0.25951195684499373
 -0.31455324374653115
 -0.8355028227748592
 -1.2201198687551253
 -1.4059382430585239
 -1.34842015730341
 -1.0544653765329255
 -0.5996547574378484
 -0.05295841933251216
  0.5051002430066458
  0.991580265343209
  1.4022161868507759
  1.4824470505184384
  1.3847897652454038
  1.0908094247111528
  0.6323706745818122
  0.08038219330185685
 -0.4801435586610426
 -0.9631055614275748
 -1.2929464955670193
 -1.4056759139512935
 -1.2565055183682223
 -0.9003403857160466
 -0.4054174735015122
  0.15291067455174231
  0.6921176042640974
  1.1303212905867048
  1.4510719377157377
  1.4540023348716802
  1.2848655917268634
  0.9295008705436955
  0.43655499378195894
 -0.12021140373032208
 -0.6559514341292595
 -1.0885179382948467
 -1.3481227295231093
 -1.368928192174748
 -1.1305361771642606
 -0.7105738857205002
 -0.1785408779461997
  0.3835025915179654
  0.8892452513285463
  1.262734570852422
  1.4807005469491064
  1.4003175339214111
  1.1536749376547422
  0.7420665530795693
  0.22005124992299324
 -0.3340748350190745
 -0.8348183797133069
 -1.2046535825239117
 -1.3803764494678368
 -1.301046097399617
 -0.9751331465967122
 -0.49074169810995927
  0.0715180089192012
  0.6246377341907545
  1.0839125912591634
  1.3821265615176461];

q = 1;
for i = 1:length(X0_vec)
    
    % polytopic initial set
    input_set  = X0_vec(i);
    poly_seq_vec   = X0_vec(i);
    poly_seq_vec2   = X0_vec(i);
    
    
    % forward reachability
    for k = 1:N
        net_p.activation = 'relu';
        [b_out,~,~] =  reach_sdp(net_p,sys.A,sys.B,input_set.H,A_out',options);
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
        input_set = Polyhedron(A_out, b_out);
        %tssos_set = Polyhedron(A_out, b_out);
        tssos_set = Polyhedron(A_out, tssos_result(q:q+(poly_size-1))); q = q + poly_size;
        
        % save results
        poly_seq_vec = [poly_seq_vec Polyhedron(A_out, b_out)];
        poly_seq_vec2 = [poly_seq_vec2 tssos_set];
        
        % report
        disp(['Reach-SDP Progress: N = ', num2str(k), ', i = ',...
            num2str(i), ', volume: ', num2str(input_set.volume)]);
    end
    poly_cell{1,i} = poly_seq_vec;
    poly_cell2{1,i} = poly_seq_vec2;
end


%% Plot results
figure('Renderer', 'painters')
%hold on
% initial set
plot(X0_poly,'color','b','alpha',0.1)
hold on
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
            hold on
        end
    end
end

% gridding states
for k = 2:N+1
    FRS = Xg_cell{k};
    FRS_bd = FRS(boundary(FRS(:,1), FRS(:,2), 0.5),:);
    plot(FRS_bd(:,1),FRS_bd(:,2),'b-','LineWidth',5)
    hold on
end

grid on;
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'));
xlim([0.9,1.5]);
ylim([-0.1,1.1]);
xlabel('$x_1$','Interpreter','latex','FontSize', 60);
ylabel('$x_2$','Interpreter','latex','FontSize', 60);
ax = gca;
ax.FontSize = 30;
% ax2 = get(gca,'XTickLabel');
% set(gca,'XTickLabel',ax2,'fontsize',22)
% ax3 = get(gca,'YTickLabel');
% set(gca,'YTickLabel',ax3,'fontsize',22)
set(gcf,'position',[0,0,1400,1000])