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
load netDoubleIntRandWeights8
net_p.dims = net2.dims;
net_p.weights = net2.weights;
net_p.biases = net2.biases;
net_p.activation = 'sigmoid';

%% Setup
% initial set
X0_b = [2.0; -1.0; 2.5; -1.5];
X0_poly = Polyhedron([1 0; -1 0; 0 1; 0 -1], X0_b);
X0 = X0_poly.outerApprox; % normalize the A matrix
X0_vec = X0;

%dx = 0.02; % shrink the tube for better visualization
dx = 0.00;
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
        x_next = A*x + B*u;
        Xg_k = [Xg_k; x_next'];
        Ug_k = [Ug_k; u(1)];
    end
    Xg_cell{end+1} = Xg_k;
    Ug_cell{end+1} = Ug_k;
end


%% Reach-SDP
poly_cell = cell(1,N+1);
poly_cell{1,1}  = X0_vec;
poly_cell2 = poly_cell;

options.verbose = 0;
options.solver = 'mosek';
options.repeated = 0;

tssos_result = [4.56689695837158
   5.22718996238762
   5.091634184490407
   4.180979763678012
   2.6337666273089186
   1.040564751007511
  -0.6661008224029782
  -1.7416143345977526
  -2.5374582131910532
  -2.9470059086875247
  -2.907841076525702
  -2.426042815601716
  -1.57489818137192
  -0.13146549038557168
   1.3784421537080125
   3.211356099363131
   7.296655363231901
   7.822574002675074
   7.15770852508379
   5.4030929933981415
   2.826640494420575
   0.18601253800802764
  -1.7308532445435896
  -3.2060952041672173
  -4.171948839219968
  -4.502667909354778
  -4.1478619376483765
  -3.1616068773836545
  -1.6940312263698252
   0.3958869046131334
   3.175166530648805
   5.659852753933344
  10.224823545122455
  10.606268696139235
  9.373103264268572
  6.713185689358754
  3.033626602742042
 -0.4608398615697429
 -2.8792411335305843
 -4.783381458756643
 -5.942843152919834
 -6.197579471510365
 -5.508721601000941
 -3.9813331411600705
 -1.8477145697357569
  1.2037349400281205
  5.095510847600311
  8.28674178704444
 13.363926899560415
 13.589630146224906
 11.74634333149793
  8.115742271487163
  3.2502829683689907
 -0.9227264455912859
 -4.127645000031221
 -6.504086217417176
 -7.876437111390137
 -8.049688946996074
 -6.99736145857691
 -4.879816829898184
 -2.019364576770992
  2.164588184013019
  7.161988696593505
  11.103865662900049
  16.720953164131604
  16.775286029203556
  14.27563283206333
   9.603178428493678
   3.469042752920486
  -1.559840547389465
  -5.492966825002278
  -8.384382778174777
  -9.987914389137448
 -10.070778048343493
  -8.620513436683773
  -5.857910734070821
  -2.20347485684724
   3.2310303196621555
   9.379606339961777
  14.121711431929242
  20.295388656385974
  20.159717860345687
  16.95473231675563
  11.168717329485654
   3.6839819815943704
  -2.265392664138965
  -6.986758510254109
 -10.43594240851377
 -12.29105985427576
 -12.274932061337989
 -10.390116968170304
  -6.923381822028458
  -2.40273008369215
   4.384513277523822
  11.750065288268157
  17.341822552590713];

q = 1;
for i = 1:length(X0_vec)
    
    % polytopic initial set
    input_set  = X0_vec(i);
    poly_seq_vec   = X0_vec(i);
    poly_seq_vec2   = X0_vec(i);
    
    
    % forward reachability
    for k = 1:N
        %net_p.activation = 'relu';
        %[b_out,~,~] =  reach_sdp(net_p,sys.A,sys.B,input_set.H,A_out',options);
        %[Y_min,Y_max,X_min,X_max,out_min,out_max] = intervalBoundPropagation(u_min,u_max,dim_hidden,net_p);
        if k == 1
        dim_x = dims(1);
        dim_u = size(B,2);

        Fx = input_set.H(:,1:end-1);
        fx = input_set.H(:,end);

        dim_px = length(fx);

        x = sdpvar(dim_x,1);
        x_min = zeros(dim_x,1);
        x_max = zeros(dim_x,1);
        options = sdpsettings('solver','mosek','verbose',0);
        for j=1:dim_x
            optimize([Fx*x<=fx],x(j),options);
            x_min(j,1) = value(x(j));

            optimize([Fx*x<=fx],-x(j),options);
            x_max(j,1) = value(x(j));
        end
        else
           x_min = min_temp;
           x_max = max_temp;
        end

        % Interval arithmetic to find the activation bounds
        %[Y_min,Y_max] = net.interval_arithmetic(x_min,x_max);

        %X_min = net.activate(Y_min);
        %X_max = net.activate(Y_max);
        [Y_min,Y_max,X_min,X_max,out_min,out_max] = intervalBoundPropagation(x_min,x_max,dims(2:end-1),net_p);
        A_temp = [1.0000     0
    0.0000    1.0000
   -1.0000    0.0000
   -0.0000   -1.0000];
        min_temp = A*x_min + B*out_min;
        max_temp = A*x_max + B*out_max;
        b_out(1) = max_temp(1);
        b_out(2) = max_temp(2);
        b_out(3) = -min_temp(1);
        b_out(4) = -min_temp(2);
        
        % shift horizon
        input_set = Polyhedron(A_temp, b_out.');
        %tssos_set = Polyhedron(A_out, b_out);
        tssos_set = Polyhedron(A_out, tssos_result(q:q+(poly_size-1))); q = q + poly_size;
        
        % save results
        poly_seq_vec = [poly_seq_vec Polyhedron(A_temp, b_out.')];
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
hold on
% initial set
plot(X0_poly,'color','b','alpha',0.1)

% N-step FRS
for i = 1:length(X0_vec)
    for k = 1:N+1
        FRS_V = poly_cell{1,i}(k).V;
        FRS_V_bd = FRS_V(boundary(FRS_V(:,1), FRS_V(:,2), 0.0),:);
        if k == 1
            continue
%             plot(FRS_V_bd(:,1),FRS_V_bd(:,2),'k-','LineWidth',2)
        else
            plot(FRS_V_bd(:,1),FRS_V_bd(:,2),'r-','LineWidth',5)
        end
    end
end

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
set(gca,'LooseInset',get(gca,'TightInset'));
xlim([0,30]);
ylim([0,6]);
xlabel('$x_1$','Interpreter','latex','FontSize', 60);
ylabel('$x_2$','Interpreter','latex','FontSize', 60);
ax = gca;
ax.FontSize = 30;
% ax2 = get(gca,'XTickLabel');
% set(gca,'XTickLabel',ax2,'fontsize',22)
% ax3 = get(gca,'YTickLabel');
% set(gca,'YTickLabel',ax3,'fontsize',22)
set(gcf,'position',[0,0,1400,1000])