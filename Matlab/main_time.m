clear;
clc;
rng(0); % Initial seed to allow reproducibility of results
addpath('followUpGenerators');
addpath('functions');

load('imagenet_resnet50_result.mat')
global confArr;


%% Configuration
numRounds = 10;  % 总共 10 轮
nRuns = 50;      % 每轮包含 50 次 run
popSize = 100;
maxGen = 250;
nvar = length(confArr);

options = optimoptions(@gamultiobj, 'PopulationType','bitstring', ...
    'PopulationSize', popSize, 'MaxGenerations', maxGen, ...
    'CrossoverFraction', 0.8);

algorithm = "NSGA-II";

%% 记录 10 轮时间
timePerRound = zeros(numRounds,1);

for rr = 1:numRounds
    fprintf('=== 开始第 %d 轮 (共 %d) ===\n', rr, numRounds);
    tic; % 开始计时整个50 run
    
    for ii=1:nRuns
        % ====== 一次 NSGA-II run ======
        [x, fval, exitflag, output, population, scores] = ...
        gamultiobj(@fitnessFunctionTestSelReverse, nvar, [],[],[],[],[], [], options);

        % ===== MR1~MR5 原始逻辑 =====
        % MR1
        failArr = failArrFlipLeftRight;
        for jj=1:size(x,1)
            failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
            numOfTestCases(1,jj)=sum(x(jj,:));
            G(jj,1) = failureDetectionCapability(1,jj);
            G(jj,2) = numOfTestCases(1,jj)/nvar;
        end
        [~, maxIndex] = max(G(:,1));
        maxIndex = min(maxIndex, size(x, 1));
        optimalSolution_MR1 = x(maxIndex, :);
        selectedTestCases_MR1 = find(optimalSolution_MR1);
        maxFailureDetectionCapability_MR1 = failureDetectionCapability(1,maxIndex);
        maxNumOfTestCases_MR1 = numOfTestCases(1,maxIndex);

        % MR2
        failArr = failArrgaussian;
        for jj=1:size(x,1)
            failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
            numOfTestCases(1,jj)=sum(x(jj,:));
            G(jj,1) = failureDetectionCapability(1,jj);
            G(jj,2) = numOfTestCases(1,jj)/nvar;
        end
        [~, maxIndex] = max(G(:,1));
        maxIndex = min(maxIndex, size(x,1));
        optimalSolution_MR2 = x(maxIndex, :);
        selectedTestCases_MR2 = find(optimalSolution_MR2);
        maxFailureDetectionCapability_MR2 = failureDetectionCapability(1,maxIndex);
        maxNumOfTestCases_MR2 = numOfTestCases(1,maxIndex);

        % MR3
        failArr = failArrRotatePlus;
        for jj=1:size(x,1)
            failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
            numOfTestCases(1,jj)=sum(x(jj,:));
            G(jj,1) = failureDetectionCapability(1,jj);
            G(jj,2) = numOfTestCases(1,jj)/nvar;
        end
        [~, maxIndex] = max(G(:,1));
        maxIndex = min(maxIndex, size(x,1));
        optimalSolution_MR3 = x(maxIndex, :);
        selectedTestCases_MR3 = find(optimalSolution_MR3);
        maxFailureDetectionCapability_MR3 = failureDetectionCapability(1,maxIndex);
        maxNumOfTestCases_MR3 = numOfTestCases(1,maxIndex);

        % MR4
        failArr = failArrcolored;
        for jj=1:size(x,1)
            failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
            numOfTestCases(1,jj)=sum(x(jj,:));
            G(jj,1) = failureDetectionCapability(1,jj);
            G(jj,2) = numOfTestCases(1,jj)/nvar;
        end
        [~, maxIndex] = max(G(:,1));
        maxIndex = min(maxIndex, size(x,1));
        optimalSolution_MR4 = x(maxIndex, :);
        selectedTestCases_MR4 = find(optimalSolution_MR4);
        maxFailureDetectionCapability_MR4 = failureDetectionCapability(1,maxIndex);
        maxNumOfTestCases_MR4 = numOfTestCases(1,maxIndex);

        % MR5
        failArr = failArrbrightness;
        for jj=1:size(x,1)
            failureDetectionCapability(1,jj)=x(jj,:)*failArr/sum(failArr);
            numOfTestCases(1,jj)=sum(x(jj,:));
            G(jj,1) = failureDetectionCapability(1,jj);
            G(jj,2) = numOfTestCases(1,jj)/nvar;
        end
        [~, maxIndex] = max(G(:,1));
        maxIndex = min(maxIndex, size(x,1));
        optimalSolution_MR5 = x(maxIndex, :);
        selectedTestCases_MR5 = find(optimalSolution_MR5);
        maxFailureDetectionCapability_MR5 = failureDetectionCapability(1,maxIndex);
        maxNumOfTestCases_MR5 = numOfTestCases(1,maxIndex);
    end

    % ======= 50次run完成，记录时间 =======
    timePerRound(rr) = toc;
    fprintf('第 %d 轮 (50 runs) 总耗时: %.4f 秒\n', rr, timePerRound(rr));
end

% 保存时间结果到 TXT
fid = fopen('/Users/miya_wang/Desktop/results/time/nsga/imagenet_resnet50_selection.txt','w');
for rr = 1:numRounds
    fprintf(fid, '%.6f\n', timePerRound(rr));
end
fclose(fid);

disp('所有轮次完成，时间已保存。');
