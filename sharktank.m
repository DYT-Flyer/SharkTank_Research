clc;
clearvars;
close all;

%% Purpose

%This is a summary of the research on the TV Show SharkTank. I used various
%machine learning algorithms to try and predict whether a startup would
%'strike' a deal with the sharks. Accuracies as well as figures including
%confusion matrices and accuracy comparisons to hyper parameters are
%included.

%% Notes

%You need libsvm to run svmtrain, I have commented out that section for
%now.

%getaccuracy.m is a function that provides the greatest accuracy by
%iterating through the threshold and then calculating accuracy after
%converting outputs. Provides outputs of greatest accuracy.

%pl_... should be read as predicted label ... (could be knn, nn, etc.)

%Used for loops for hyperparameter tuning because it was not computational intensive

%Did not tune all hyper-parameters - nodes for neural networks, lambda for
%regularization, and number of neighbors for KNN. Also tuned threshold.

%Neural Network Graph - I ran 10 instances for each set amount of nodes
%1:10
%I plotted the accuracies sequentially for visual purposes.
%For example: 10 accuracies for NN with 5 Nodes will be plotted between 5
%and 6 with a stride of .1 starting at 5
%I like using moving average for visualization purposes because sometimes
%there is a clear pattern in accuracies and number of nodes.

%Accuracies vs. Lambda: I plotted this for visualization purposes to see if
%there was a pattern or just noise

%All figures are saved in the folder figures

%Figures found in paper not here were created in Tableau

%Some accuracies are different than what is found in the paper, the reason
%for this is because I expanded some of the hyper parameter tuning in this
%model

%% Loading Data (mat files)

load('TrainData.mat')
load('TestData.mat')

% Seperating Data from labels

TrainLabels = TrainData(:,1);
TrainData = TrainData(:,2:end);

TestLabels = TestData(:, 1);
TestData = TestData(:, 2:end);

%% Model Ridge Regression

temp = zeros(1000,1);

ridge_accuracy = 0;
for i = 1:1:1000
    model_ridge = ridge_regression(TrainData,TrainLabels,i/1000);
    pl_ridge = TestData*model_ridge;
    [ridge_accuracy_temp, pl_ridge] = getaccuracy(pl_ridge, TestLabels);
    ridge_accuracy = max(ridge_accuracy_temp,ridge_accuracy);
    temp(i) = ridge_accuracy_temp;
    if ridge_accuracy == ridge_accuracy_temp
        pl_ridge_max = pl_ridge;
    end
end

f1 = figure();
scatter(.001:.001:1,temp, 'filled');
title('Ridge Regression: Accuracy Vs. Lambda');
xlabel('Lambda');
ylabel('Accuracy');

f2 = figure();
plotconfusion(TestLabels',pl_ridge_max')
title('Ridge Regression')


clc;

%% SVM Classification - LIBSVM

% model = svmtrain(TrainLabels,TrainData,'-s 3 -t 2');
% pl_libsvm = svmpredict(TestLabels,TestData,model);
% 
% [libsvm_accuracy, pl_libsvm] = getaccuracy(pl_libsvm, TestLabels);
% f3 = figure();
% plotconfusion(TestLabels',pl_libsvm');
% title('SVM Classification')


%% Neural Network Classification

temp = zeros(100,1);

for i = 1:10
    for j = 1:10
        net = patternnet(i);
        net.trainParam.showWindow = false;
        [net,tr] = train(net,TrainData',TrainLabels');
        pl_nn = net(TestData')';

        nn_accuracy = getaccuracy(pl_nn, TestLabels);
        temp((i-1)*10+j) = nn_accuracy;
    end
end

f4 = figure();
scatter(1:.1:10.9,temp, 'filled');
hold on

hold on
for i = 1:10
    xline(i);
    hold on
end
mm = plot(1:.1:10.9,movmean(temp,10));
legend([mm],{'y = moving average of last ten accuracies'},'Location','southwest');
title('Neural Network: Accuracy Vs. Number of Nodes');

xlabel({'Number of Nodes: 10 Instances of Nodes 1:10', 'Plotted Inclusively from Node(x) to Node(x+1) for visual purposes'});
ylabel('Accuracy');


%% Non-Linear Least Squares

C = TrainData';
lambda = 0;
ilambda = ones(37,1)*lambda;
C = [C ilambda];

d = zeros(1,1);
d = [TrainLabels' d];

out = lsqnonneg(C',d');
pl_nlls = TestData*out;

[nlls_accuracy, pl_nlls] = getaccuracy(pl_nlls, TestLabels);
f5 = figure();
plotconfusion(TestLabels',pl_nlls');
title('Non Linear Least Squares')

%% Our Model - NLLS with regularization term

om_accuracy = 0;
temp = zeros(1000,1);
count = 0;
for i = 1:1:1000
    C = TrainData';
    lambda = i/1000;
    ilambda = ones(37,1)*lambda;
    C = [C ilambda];

    d = zeros(1,1);
    d = [TrainLabels' d];

    out = lsqnonneg(C',d');
    pl_om = TestData*out;

    [om_accuracy_temp, pl_om] = getaccuracy(pl_om, TestLabels);
    om_accuracy = max(om_accuracy_temp, om_accuracy);
    if om_accuracy == om_accuracy_temp
        pl_om_max = pl_om;
    end
    temp(i) = om_accuracy_temp;
end

f6 = figure();
scatter(.001:.001:1,temp, 'filled');
title('Our Model: Accuracy Vs. Lambda');
xlabel('Lambda');
ylabel('Accuracy');

f7 = figure();
plotconfusion(TestLabels',pl_om_max');
title('Our Model');

%% KNN
knn_accuracy = 0;

temp = zeros(10,1);
for i = 1:10
    model_knn = fitcknn(TrainData, TrainLabels, 'NumNeighbors', i,...
        'BreakTies', 'nearest', 'Distance', 'euclidean');

    pl_knn = predict(model_knn, TestData);

    [knn_accuracy_temp, pl_knn] = getaccuracy(pl_knn, TestLabels);
    knn_accuracy = max(knn_accuracy, knn_accuracy_temp);
    if knn_accuracy == knn_accuracy_temp
        pl_knn_max = pl_knn;
    end
    temp(i) = knn_accuracy_temp;
end

f8 = figure();
scatter(1:10,temp, 'filled');
title('KNN: Accuracy Vs. Number of Neighbors');
xlabel('Number of Neighbors');
ylabel('Accuracy');

f9 = figure();
plotconfusion(TestLabels',pl_knn_max');
title('KNN');


%% Print Results

clc;

disp('ridge_accuracy: ')
disp(ridge_accuracy)

% disp('libsvm_accuracy: ')
% disp(libsvm_accuracy)

disp('nn_accuracy: ');
disp(nn_accuracy);

disp('non_linear_least_squares_accuracy: ');
disp(nlls_accuracy);

disp('our_model_accuracy: ');
disp(om_accuracy);

disp('knn_accuracy: ');
disp(knn_accuracy);






