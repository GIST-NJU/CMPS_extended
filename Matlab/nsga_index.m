clear;
clc;
rng(0); % Initial seed to allow reproducibility of results
addpath('followUpGenerators');
%addpath('pareto_HV');
addpath('functions');

load('fruit_shuffle_result.mat')
global confArr;

%% Experiment variables

expResults{1,1} = 'Algorithm';
expResults{1,2} = 'MR1_FailureDetectionCapability';
expResults{1,3} = 'MR1_NumOfTestCases';
expResults{1,4} = 'MR1_optimalSolution';

expResults{1,5} = 'MR2_FailureDetectionCapability';
expResults{1,6} = 'MR2_NumOfTestCases';
expResults{1,7} = 'MR2_optimalSolution';

expResults{1,8} = 'MR3_FailureDetectionCapability';
expResults{1,9} = 'MR3_NumOfTestCases';
expResults{1,10} = 'MR3_optimalSolution';

expResults{1,11} = 'MR4_FailureDetectionCapability';
expResults{1,12} = 'MR4_NumOfTestCases';
expResults{1,13} = 'MR4_optimalSolution';

expResults{1,14} = 'MR5_FailureDetectionCapability';
expResults{1,15} = 'MR5_NumOfTestCases';
expResults{1,16} = 'MR5_optimalSolution';

%% Configuration of the experimental set-up
nRuns = 10;
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
   maxIndex = min(maxIndex, size(x, 1)); 
   optimalSolution_MR1 = x(maxIndex, :);
   selectedTestCases_MR1 = find(optimalSolution_MR1);
   
   % 找到对应的 failureDetectionCapability 和 numOfTestCases
   maxFailureDetectionCapability_MR1 = failureDetectionCapability(1,maxIndex);
   maxNumOfTestCases_MR1 = numOfTestCases(1,maxIndex);
   
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
   maxIndex = min(maxIndex, size(x,1));
   optimalSolution_MR2 = x(maxIndex, :);
   selectedTestCases_MR2 = find(optimalSolution_MR2);
   
   % 找到对应的 failureDetectionCapability 和 numOfTestCases
   maxFailureDetectionCapability_MR2 = failureDetectionCapability(1,maxIndex);
   maxNumOfTestCases_MR2 = numOfTestCases(1,maxIndex);
   
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
   maxIndex = min(maxIndex, size(x,1));
   optimalSolution_MR3 = x(maxIndex, :);
   selectedTestCases_MR3 = find(optimalSolution_MR3);
   
   % 找到对应的 failureDetectionCapability 和 numOfTestCases
   maxFailureDetectionCapability_MR3 = failureDetectionCapability(1,maxIndex);
   maxNumOfTestCases_MR3 = numOfTestCases(1,maxIndex);

   
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
   maxIndex = min(maxIndex, size(x,1));
   optimalSolution_MR4 = x(maxIndex, :);
   selectedTestCases_MR4 = find(optimalSolution_MR4);

   % 找到对应的 failureDetectionCapability 和 numOfTestCases
   maxFailureDetectionCapability_MR4 = failureDetectionCapability(1,maxIndex);
   maxNumOfTestCases_MR4 = numOfTestCases(1,maxIndex);
   
   
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
   maxIndex = min(maxIndex, size(x,1));
   optimalSolution_MR5 = x(maxIndex, :);
   selectedTestCases_MR5 = find(optimalSolution_MR5);
   
   % 找到对应的 failureDetectionCapability 和 numOfTestCases
   maxFailureDetectionCapability_MR5 = failureDetectionCapability(1,maxIndex);
   maxNumOfTestCases_MR5 = numOfTestCases(1,maxIndex);
   
    expResults{ii,1} = 'NSGA-II';
    expResults{ii,2} = maxFailureDetectionCapability_MR1;
    expResults{ii,3} = maxNumOfTestCases_MR1;
    expResults{ii,4} = selectedTestCases_MR1;

    expResults{ii,5} = maxFailureDetectionCapability_MR2;
    expResults{ii,6} = maxNumOfTestCases_MR2;
    expResults{ii,7} = selectedTestCases_MR2;

    expResults{ii,8} = maxFailureDetectionCapability_MR3;
    expResults{ii,9} = maxNumOfTestCases_MR3;
    expResults{ii,10} = selectedTestCases_MR3;

    expResults{ii,11} = maxFailureDetectionCapability_MR4;
    expResults{ii,12} = maxNumOfTestCases_MR4;
    expResults{ii,13} = selectedTestCases_MR4;

    expResults{ii,14} = maxFailureDetectionCapability_MR5;
    expResults{ii,15} = maxNumOfTestCases_MR5;
    expResults{ii,16} = selectedTestCases_MR5;

end

 %xlswrite('NSGAII_Reverse_GOOGLENet.xlsx',expResults);
 disp(expResults);

 mr1_results = cell2mat(expResults(:, 2));
[minValue_mr1, minIndex_mr1] = min(mr1_results);
% 根据索引找到对应的 selectedTestCases_MR1
selectArray_mr1 = expResults{minIndex_mr1, 4}; 
% 保存结果到 .mat 文件
save('fruit360_shufflenet_mr1.mat', 'selectArray_mr1');

 mr2_results = cell2mat(expResults(:, 5));
[minValue_mr2, minIndex_mr2] = min(mr2_results);
selectArray_mr2 = expResults{minIndex_mr2, 7}; 
save('fruit360_shufflenet_mr2.mat', 'selectArray_mr2');

 mr3_results = cell2mat(expResults(:, 8));
[minValue_mr3, minIndex_mr3] = min(mr3_results);
selectArray_mr3 = expResults{minIndex_mr3, 10}; 
save('fruit360_shufflenet_mr3.mat', 'selectArray_mr3');

 mr4_results = cell2mat(expResults(:, 11));
[minValue_mr4, minIndex_mr4] = min(mr4_results);
selectArray_mr4 = expResults{minIndex_mr4, 13}; 
save('fruit360_shufflenet_mr4.mat', 'selectArray_mr4');

 mr5_results = cell2mat(expResults(:, 14));
[minValue_mr5, minIndex_mr5] = min(mr5_results);
selectArray_mr5 = expResults{minIndex_mr5, 16}; 
save('fruit360_shufflenet_mr5.mat', 'selectArray_mr5');


