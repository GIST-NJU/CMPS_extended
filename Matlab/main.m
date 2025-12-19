clear;
clc;
rng(0); % Initial seed to allow reproducibility of results
addpath('followUpGenerators');
%addpath('pareto_HV');
addpath('functions');

%load('imageresnet.mat')
load('mnistlenet5_result.mat')
global confArr;
%data = load('mnistlenet1_result.mat'); % 以结构体方式加载
%fieldnames(data) % 查看变量列表

%% Experiment variables

expResults{1,1} = 'Algorithm';
expResults{1,2} = 'MR1_FailureDetectionCapability';
expResults{1,3} = 'MR1_NumOfTestCases';
expResults{1,4} = 'MR1_AverageFailureDetectionRate';
expResults{1,5} = 'optimalSolution_MR1';

expResults{1,6} = 'MR2_FailureDetectionCapability';
expResults{1,7} = 'MR2_NumOfTestCases';
expResults{1,8} = 'MR2_AverageFailureDetectionRate';
expResults{1,9} = 'optimalSolution_MR2';

expResults{1,10} = 'MR3_FailureDetectionCapability';
expResults{1,11} = 'MR3_NumOfTestCases';
expResults{1,12} = 'MR3_AverageFailureDetectionRate';
expResults{1,13} = 'optimalSolution_MR3';

expResults{1,14} = 'MR4_FailureDetectionCapability';
expResults{1,15} = 'MR4_NumOfTestCases';
expResults{1,16} = 'MR4_AverageFailureDetectionRate';
expResults{1,17} = 'optimalSolution_MR4';

expResults{1,18} = 'MR5_FailureDetectionCapability';
expResults{1,19} = 'MR5_NumOfTestCases';
expResults{1,20} = 'MR5_AverageFailureDetectionRate';
expResults{1,21} = 'optimalSolution_MR5';

expResults{1,22} = 'Running time';

times = zeros(1, 6);

%% Configuration of the experimental set-up
for i =1:1
    tic;
    nRuns = 3;
    popSize = 100;
    maxGen = 250;
    
    nvar = length(confArr);
    
    options = optimoptions(@gamultiobj, 'PopulationType','bitstring', 'PopulationSize', ...
        popSize, 'MaxGenerations', maxGen, 'CrossoverFraction', 0.8);% ...
        %'InitialPopulationMatrix', Balanced);
    algorithm = "NSGA-II";
    failArr = failArrFlipLeftRight;
    for ii=1:nRuns
    
       %% Execute algorithm
       tic;
       [x, fval, exitflag, output, population, scores] = ...
            gamultiobj(@fitnessFunctionTestSelReverse, nvar, [],[],[],[],[], [], options);
       time = toc; 
    
       %% Obtain metrics for MR 1 - Flip left right
       failArr = failArrFlipLeftRight;
       for jj=1:size(x,1)
           failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
           numOfTestCases(1,jj)=sum(x(jj,:));
    
           % 将 failureDetectionCapability 和 numOfTestCases 存入 G
           G(jj,1) = failureDetectionCapability(1,jj);
           G(jj,2) = numOfTestCases(1,jj)/nvar;
       end
    
       % 找到 G 中 failureDetectionCapability 的最大值和对应的索引
       [~, maxIndex] = max(G(:,1));
    
       % 找到最优解对应的解决方案
       optimalSolution_MR1 = x(maxIndex, :);
    
       % 找到对应的 failureDetectionCapability 和 numOfTestCases
       maxFailureDetectionCapability_MR1 = failureDetectionCapability(1,maxIndex);
       maxNumOfTestCases_MR1 = numOfTestCases(1,maxIndex);
    
       % 随机选择最优解中的5%个测试用例并计算多次平均找错率
       numIterations = 10;  % 设定要重复的次数
       averageFailureDetectionRates_mr1 = zeros(numIterations, 1);  % 用于存储每次的平均找错率
       optimalSolution = x(maxIndex, :);
       numOptimalTestCases = sum(optimalSolution);  % 计算最优解测试用例的总数
       numRandomTests = 1000;
    
       for iter = 1:numIterations
           selectedTestCases = randsample(find(optimalSolution), numRandomTests);
           % 计算选定测试用例的找错率
           selectedFailureDetectionRate = sum(failArr(selectedTestCases)) / numRandomTests;
    
           % 存储本次的找错率
           averageFailureDetectionRates_mr1(iter) = selectedFailureDetectionRate;
       end
    
       % 计算多次平均找错率
       meanFailureDetectionRate_MR1 = mean(averageFailureDetectionRates_mr1);
    
       %% Obtain metrics for MR 2 - Flip up down
       failArr = failArrgaussian;
       for jj=1:size(x,1)
           failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
           numOfTestCases(1,jj)=sum(x(jj,:));
    
           % 将 failureDetectionCapability 和 numOfTestCases 存入 G
           G(jj,1) = failureDetectionCapability(1,jj);
           G(jj,2) = numOfTestCases(1,jj)/nvar;
       end
    
       % 找到 G 中 failureDetectionCapability 的最大值和对应的索引
       [~, maxIndex] = max(G(:,1));
    
       % 找到最优解对应的解决方案
       optimalSolution_MR2 = x(maxIndex, :);
    
       % 找到对应的 failureDetectionCapability 和 numOfTestCases
       maxFailureDetectionCapability_MR2 = failureDetectionCapability(1,maxIndex);
       maxNumOfTestCases_MR2 = numOfTestCases(1,maxIndex);
    
       % 随机选择最优解中的5%个测试用例并计算多次平均找错率
       averageFailureDetectionRates_mr2 = zeros(numIterations, 1);  % 用于存储每次的平均找错率
       optimalSolution = x(maxIndex, :);
       numOptimalTestCases = sum(optimalSolution);  % 计算最优解测试用例的总数
       numRandomTests = 1000;
    
       for iter = 1:numIterations
           selectedTestCases = randsample(find(optimalSolution), numRandomTests);
           % 计算选定测试用例的找错率
           selectedFailureDetectionRate = sum(failArr(selectedTestCases)) / numRandomTests;
    
           % 存储本次的找错率
           averageFailureDetectionRates_mr2(iter) = selectedFailureDetectionRate;
       end
    
       % 计算多次平均找错率
       meanFailureDetectionRate_MR2 = mean(averageFailureDetectionRates_mr2);
    
       %% Obtain metrics for MR 3 - Rotate minus
       failArr = failArrRotatePlus;
       for jj=1:size(x,1)
           failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
           numOfTestCases(1,jj)=sum(x(jj,:));
    
           % 将 failureDetectionCapability 和 numOfTestCases 存入 G
           G(jj,1) = failureDetectionCapability(1,jj);
           G(jj,2) = numOfTestCases(1,jj)/nvar;
       end
    
       % 找到 G 中 failureDetectionCapability 的最大值和对应的索引
       [~, maxIndex] = max(G(:,1));
    
       % 找到最优解对应的解决方案
       optimalSolution_MR3 = x(maxIndex, :);
    
       % 找到对应的 failureDetectionCapability 和 numOfTestCases
       maxFailureDetectionCapability_MR3 = failureDetectionCapability(1,maxIndex);
       maxNumOfTestCases_MR3 = numOfTestCases(1,maxIndex);
    
       % 随机选择最优解中的5%个测试用例并计算多次平均找错率
       averageFailureDetectionRates_mr3 = zeros(numIterations, 1);  % 用于存储每次的平均找错率
       optimalSolution = x(maxIndex, :);
       numOptimalTestCases = sum(optimalSolution);  % 计算最优解测试用例的总数
       numRandomTests = 500;
    
       for iter = 1:numIterations
           selectedTestCases = randsample(find(optimalSolution), numRandomTests);
           % 计算选定测试用例的找错率
           selectedFailureDetectionRate = sum(failArr(selectedTestCases)) / numRandomTests;
    
           % 存储本次的找错率
           averageFailureDetectionRates_mr3(iter) = selectedFailureDetectionRate;
       end
    
       % 计算多次平均找错率
       meanFailureDetectionRate_MR3 = mean(averageFailureDetectionRates_mr3);
    
       %% Obtain metrics for MR 4 - Rotate Plus
       failArr = failArrcolored;
       for jj=1:size(x,1)
           failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
           numOfTestCases(1,jj)=sum(x(jj,:));
    
           % 将 failureDetectionCapability 和 numOfTestCases 存入 G
           G(jj,1) = failureDetectionCapability(1,jj);
           G(jj,2) = numOfTestCases(1,jj)/nvar;
       end
    
       % 找到 G 中 failureDetectionCapability 的最大值和对应的索引
       [~, maxIndex] = max(G(:,1));
    
       % 找到最优解对应的解决方案
       optimalSolution_MR4 = x(maxIndex, :);
    
       % 找到对应的 failureDetectionCapability 和 numOfTestCases
       maxFailureDetectionCapability_MR4 = failureDetectionCapability(1,maxIndex);
       maxNumOfTestCases_MR4 = numOfTestCases(1,maxIndex);
    
       % 随机选择最优解中的5%个测试用例并计算多次平均找错率
       averageFailureDetectionRates_mr4 = zeros(numIterations, 1);  % 用于存储每次的平均找错率
       optimalSolution = x(maxIndex, :);
       numOptimalTestCases = sum(optimalSolution);  % 计算最优解测试用例的总数
       numRandomTests = 1000;
    
       for iter = 1:numIterations
           selectedTestCases = randsample(find(optimalSolution), numRandomTests);
           % 计算选定测试用例的找错率
           selectedFailureDetectionRate = sum(failArr(selectedTestCases)) / numRandomTests;
    
           % 存储本次的找错率
           averageFailureDetectionRates_mr4(iter) = selectedFailureDetectionRate;
       end
    
       % 计算多次平均找错率
       meanFailureDetectionRate_MR4 = mean(averageFailureDetectionRates_mr4);
    
       %% Obtain metrics for MR 5 - Shear
       failArr = failArrbrightness;
       for jj=1:size(x,1)
           failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
           numOfTestCases(1,jj)=sum(x(jj,:));
    
           % 将 failureDetectionCapability 和 numOfTestCases 存入 G
           G(jj,1) = failureDetectionCapability(1,jj);
           G(jj,2) = numOfTestCases(1,jj)/nvar;
       end
    
       % 找到 G 中 failureDetectionCapability 的最大值和对应的索引
       [~, maxIndex] = max(G(:,1));
    
       % 找到最优解对应的解决方案
       optimalSolution_MR5 = x(maxIndex, :);
    
       % 找到对应的 failureDetectionCapability 和 numOfTestCases
       maxFailureDetectionCapability_MR5 = failureDetectionCapability(1,maxIndex);
       maxNumOfTestCases_MR5 = numOfTestCases(1,maxIndex);
    
       % 随机选择最优解中的5%个测试用例并计算多次平均找错率
       averageFailureDetectionRates_mr5 = zeros(numIterations, 1);  % 用于存储每次的平均找错率
       optimalSolution = x(maxIndex, :);
       numOptimalTestCases = sum(optimalSolution);  % 计算最优解测试用例的总数
       numRandomTests = 1000;
    
       for iter = 1:numIterations
    
           selectedTestCases = randsample(find(optimalSolution), numRandomTests);
    
           % 计算选定测试用例的找错率
           selectedFailureDetectionRate = sum(failArr(selectedTestCases)) / numRandomTests;
    
           % 存储本次的找错率
           averageFailureDetectionRates_mr5(iter) = selectedFailureDetectionRate;
       end
    
       % 计算多次平均找错率
       meanFailureDetectionRate_MR5 = mean(averageFailureDetectionRates_mr5);
    
        expResults{ii,1} = 'NSGA-II';
        expResults{ii,2} = maxFailureDetectionCapability_MR1;
        expResults{ii,3} = maxNumOfTestCases_MR1;
        expResults{ii,4} = meanFailureDetectionRate_MR1;
        expResults{ii,5} = optimalSolution_MR1;
    
        expResults{ii,6} = maxFailureDetectionCapability_MR2;
        expResults{ii,7} = maxNumOfTestCases_MR2;
        expResults{ii,8} = meanFailureDetectionRate_MR2;
        expResults{ii,9} = optimalSolution_MR2;
    
        expResults{ii,10} = maxFailureDetectionCapability_MR3;
        expResults{ii,11} = maxNumOfTestCases_MR3;
        expResults{ii,12} = meanFailureDetectionRate_MR3;
        expResults{ii,13} = optimalSolution_MR3;
    
        expResults{ii,14} = maxFailureDetectionCapability_MR4;
        expResults{ii,15} = maxNumOfTestCases_MR4;
        expResults{ii,16} = meanFailureDetectionRate_MR4;
        expResults{ii,17} = optimalSolution_MR4;
    
        expResults{ii,18} = maxFailureDetectionCapability_MR5;
        expResults{ii,19} = maxNumOfTestCases_MR5;
        expResults{ii,20} = meanFailureDetectionRate_MR5;
        expResults{ii,21} = optimalSolution_MR5;
    
        expResults{ii,22} = time;
        R{ii} = x; % result of the array if further analysis is required
    end
    
    %xlswrite('NSGAII_Reverse_GOOGLENet.xlsx',expResults);
    disp(expResults);
    times(i) = toc;
end
%disp(times);
