% Dimensions
dim_in = 2;
dim_hidden = [5*ones(1,2)];
dim_out = 1;

sys.uub =  2;
sys.ulb = 1;
load nnmpc_nets_di_1

% Activation function type. Can be relu, sigmoid or tanh
AF = 'relu'; 

% Create NN parameters
dims = [dim_in, dim_hidden, dim_out];
%net = NNsetup(dims,AF);
%mat2JuliaNetUPDATED(net) % save the net to be used in other software

% Find max matrix size
max_dim = max(dims);

% Add projection layer
net_p = nnsequential([dims dims(end) dims(end)],'relu');

weights_p = weights;
weights_p{end+1} = -eye(dim_out);
weights_p{end+1} = -eye(dim_out);

biases_p = biases;
biases_p{end} = biases{end}-sys.ulb;
biases_p{end+1} =  sys.uub-sys.ulb;
biases_p{end+1} =  sys.uub;

net_p.weights = weights_p;
net_p.biases = biases_p;

% Create weights matrix
weights2 = zeros(max_dim,max_dim,length(dims) - 1);
biases2 = zeros(max_dim,length(dims)-1);

% Put in format for Julia
dims = net_p.dims;
weights = net_p.weights;
biases = net_p.biases;

dims = dims(1:end-2);
net = NNsetup(dims,AF);

%% 

% Find max matrix size
max_dim = max(dims);

% Add projection layer
net_p = nnsequential([dims dims(end) dims(end)],'relu');

weights_p = net.weights;
weights_p{end+1} = -eye(dim_out);
weights_p{end+1} = -eye(dim_out);

biases_p = net.biases;
biases_p{end} = biases{end}-sys.ulb;
biases_p{end+1} =  sys.uub-sys.ulb;
biases_p{end+1} =  sys.uub;

net_p.weights = weights_p;
net_p.biases = biases_p;
weights = net_p.weights;
biases = net_p.biases;
dims = net_p.dims;

%% 
%weights = net.weights;
%biases = net.biases;

for i = 1:(length(dims) - 1)
    weights2(1:dims(i+1), 1:dims(i), i) = weights{i};
    biases2(1:dims(i+1), i) = biases{i};
end

weights = weights2;
biases = biases2;
net2 = net;
net2 = net_p;

%filename = 'ReachSparsePsatz/netDoubleIntRandWeights9.mat';
filename = 'ReachSparsePsatz/netInvertedPendulum.mat';
save(filename,'weights','biases','dims','AF','net2')

%filename = sprintf('ReachSparsePsatz/net%dx%dx%dx%d%s.mat',dims(1),dims(2),length(dims)-2,dims(end),AF);
%filenameNET = sprintf('NNPsatzChor/matlabNNSavesTRUE/NETnet%dx%dx%dx%d%s.mat',dims(1),dims(2),length(dims)-2,dims(end),AF);

return
% if all(dims(2:end-1) == dims(2)) 
%     filename = sprintf('NNPsatzChor/matlabNNSavesTRUE/net%dx%dx%dx%d%s.mat',dims(1),dims(2),length(dims)-2,dims(end),AF);
%     filenameNET = sprintf('NNPsatzChor/matlabNNSavesTRUE/NETnet%dx%dx%dx%d%s.mat',dims(1),dims(2),length(dims)-2,dims(end),AF);
%     if isfile(filename)
%         % DO NOTHING
%     else
%         save(filename,'weights','biases','dims','AF')
%         save(filenameNET,'net')
%     end
%     %save('NNPsatzChor/matlabNNSaves/test.mat','weights','biases')
% else
%     %error('fix this later')
%     filename = sprintf('NNPsatzChor/matlabNNSaves/net%dx%dx%dx%dALT.mat',dims(1),dims(2),length(dims)-2,dims(end));
%     save(filename,'weights','biases','dims','AF')
% end