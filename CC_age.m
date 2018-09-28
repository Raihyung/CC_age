% Credit: adapted from Finn et al., 2015, Shen et al., 2017 

%% Internal Validation: find connectivity features predicting cognitive performance of group 1 (say, adolescents)  

load('group1_mats.mat')  % Connectivity matrices are stored in a 3D matrix of size M x M x N (M is the number of nodes, N is the number of subjects)
load('group1_behav.mat')

no_sub = size(group1_mats,3); 
no_node = size(group1_mats,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);

thresh = 0.05;  % threshold for feature selection

for leftout = 1:no_sub;
    fprintf('\n Leaving out subj # %6.3f', leftout);
    
    % leave out subject from matrices and behavior 
    train_mats = group1_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats, [], size(train_mats,3));
    
    train_behav = group1_behav;
    train_behav(leftout) =[];
    
    % correlate all edges with behavior
    [r_mat,p_mat] = corr(train_vcts',train_behav);
    r_mat = reshape(r_mat,no_node,no_node);
    p_mat = reshape(p_mat,no_node,no_node);
    
    % set threshold and define masks
    pos_mask = zeros(no_node, no_node);
    neg_mask = zeros(no_node, no_node);
    
    pos_edges = find(r_mat > 0 & p_mat < thresh);
    neg_edges = find(r_mat < 0 & p_mat < thresh);
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1; 
            
    pos_overlap(:,:,leftout) = pos_mask;    % find the set of edges that appeared in the positive network of every iteration of the leave-one-out cross-validation
    neg_overlap(:,:,leftout) = neg_mask;    % find the set of edges that appeared in the negative network of every iteration of the leave-one-out cross-validation
     
    % get sum of all edges in TRAIN subs 
    train_sumpos = zeros(no_sub-1, 1);
    train_sumneg = zeros(no_sub-1, 1);
    
    for ss = 1:size(train_sumpos);
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
    end
    
    % build model on TRAIN subs
    fit_pos = polyfit(train_sumpos, train_behav, 1);
    fit_neg = polyfit(train_sumneg, train_behav, 1);
    
    % run model on TEST sub
    test_mat = group1_mats(:,:,leftout);
    test_sumpos = sum(sum(test_mat.*pos_mask))/2;
    test_sumneg = sum(sum(test_mat.*neg_mask))/2;
    
   behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
   behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2); 
end

% compare predicted and observed scores
[R_pos, P_pos] = corr(behav_pred_pos, group1_behav)
[R_neg, P_neg] = corr(behav_pred_neg, group1_behav)

figure(1); plot(behav_pred_pos, group1_behav, 'r.'); lsline
figure(2); plot(behav_pred_neg, group1_behav, 'b.'); lsline



%% External Prediction: use connectivity features found in group 1 to predict cognitive performance of group 2 (say, young adults)

load('group2_mats.mat')  
load('group2_behav.mat')

% find the set of edges that appeared in the positive/negative network of 90% of iterations of the LOOCV
P_mask = zeros(no_node, no_node);
N_mask = zeros(no_node, no_node);

if R_pos > 0 && P_pos < 0.05     % use the network that was internally validated 
    P_sum = sum(pos_overlap,3); 
    P_edges = find(P_sum > no_sub*0.9);    
    P_mask(P_edges) = 1;
end

if R_neg > 0 && P_neg < 0.05     % use the network that was internally validated 
    N_sum = sum(neg_overlap,3); 
    N_edges = find(N_sum > no_sub*0.9);    
    N_mask(N_edges) = 1;
end
    
% apply model to group 2
for i = 1:size(group2_mats,3)
    P_pred_group2(i) = sum(sum(group2_mats(:,:,i).*P_mask))/2;
    N_pred_group2(i) = sum(sum(group2_mats(:,:,i).*N_mask))/2;
end

% compare predicted and observed scores
[R_pos_group2, P_pos_group2] = corr(P_pred_group2', group2_behav)
[R_neg_group2, P_neg_group2] = corr(N_pred_group2', group2_behav)

figure(3); plot(P_pred_group2', group2_behav, 'r.'); lsline
figure(4); plot(N_pred_group2', group2_behav, 'b.'); lsline



%% Visualize selected connectivity features by functional networks to which nodes belong (here, Power 264 ROI used; Power et al., 2011)

% creat network mask cells (uncertain nodes excluded)
a{1} = [1:35]; % Sensory 
a{2} = [36:49, 182:199]; % 36:49 Cinguloopercular / 182:199 salience
a{3} = [50:62];% Audi
a{4} = [63:125];% 63:120 DMN 121:125 Memory
a{5} = [126:156];% Vis
a{6} = [157:181];% FP
a{7} = [200:212];% Subcort
a{8} = [213:232];% VAN 213:221 / DAN 222:232
a{9} = [233:236];% Cerebellar

no_nets = length(a);

% assign each network mask into m by n cell
cell = {};
for m = 1:no_nets
    for n =1:no_nets
%         cell{m,n} = P_mask(a{m}, a{n});  
        cell{m,n} = N_mask(a{m}, a{n});    
    end
end

% sum within each cell
cell2=zeros(no_nets,no_nets);
for m = 1:no_nets
    for n = 1:no_nets
        cell2(m,n)=sum(sum(cell{m,n}));
    end
end

% mean within each cell
cell3=zeros(no_nets,no_nets);
for m = 1:no_nets
    for n = 1:no_nets
        cell3(m,n)=mean(mean(cell{m,n}))
    end
end

% powermat = tril(cell2);   % Select sum
powermat = tril(cell3);   % Select mean

figure(5); imagesc(powermat);      % Create a colored plot of the matrix values
colormap(flipud(gray));  % Change the colormap to gray (so higher values are
                         %   black and lower values are white)

textStrings = num2str(powermat(:), '%0.2f');       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
[x, y] = meshgrid(1:no_nets);  % Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range
textColors = repmat(powermat(:) > midValue, 1, 3);  % Choose white or black for the text color of the strings so
                                                    % they can be easily seen over the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors
set(gca, 'XTick', 1:no_nets, ...                             % Change the axes tick marks
         'XTickLabel', {'Sen', 'CO', 'Aud', 'DMN', 'Vis', 'FP', 'Sub', 'Att', 'CBN'}, ...  %   and tick labels
         'YTick', 1:no_nets, ...
         'YTickLabel', {'Sen', 'CO', 'Aud', 'DMN', 'Vis', 'FP', 'Sub', 'Att', 'CBN'}, ...
         'TickLength', [0 0]);


%% References
% Finn, E.S. et al. Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity. Nat. Neurosci. 18, 1664-1671 (2015).
% Shen, X. et al. Using connectome-based predictive modeling to predict individual behavior from brain connectivity. Nat. Protoc. 12(3), 506-518 (2017). 
% Power, J.D. et al. Functional network organization of the human brain. Neuron 72, 665-678 (2011).
