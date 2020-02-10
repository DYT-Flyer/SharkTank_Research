clc;
clearvars;
close all;

%% Calculated Correlation Coefficients

load('TrainData.mat');
ccs = zeros(1,37);
for i = 1:37
    ccs(1,i) = corrcoef(TrainData(:,1),TrainData(:,i+1));
end
