function [maxacc,pl] = getaccuracy(preds, labels)
%% Iterates through threshold to provide greatest accuracy, returns predicted results of greatest accuracy

[n,~] = size(labels);
maxacc = 0;
for i = 0:.01:1
    predictedLabel = preds;
    predictedLabel(preds>= i) = 1;
    predictedLabel(preds< i) = 0;
    accurateClassification = 0;
    
    for k = 1:n
        if(predictedLabel(k) == labels(k,1))
            accurateClassification = accurateClassification + 1;
        end
    end
    
    acc = accurateClassification/n;
    maxacc = max(acc,maxacc);
    if acc==maxacc
        pl = predictedLabel;
    end
end
end
