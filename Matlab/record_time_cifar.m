%% 清空环境
clear; clc; close all;

%% 设置参数
testDir = '/Users/miya_wang/Desktop/CMPS++/cifar-10-batches-py/test_img';  % CIFAR-10 测试集路径
numRounds = 10;

outputFiles = { ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/cmps/cifar10_resnet50.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/cmps++/cifar10_resnet50.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/nsga/cifar10_resnet50.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/mpss/cifar10_resnet50.txt' ...
};

%% 读取 CIFAR-10 测试图像
imdsTest = imageDatastore(testDir, 'IncludeSubfolders', true, 'LabelSource', 'none');
numImages = numel(imdsTest.Files);
fprintf('测试集包含 %d 张图片\n', numImages);

%% 加载 CIFAR-10 VGG19 模型
load('CIFAR10ResNet50.mat', 'netTransfer');  % 确认变量名为 net
% load('CIFAR10VGG19.mat', 'netTransfer');

%% 获取输入大小
inputSize = netTransfer.Layers(1).InputSize;   % 应该是 [32 32 3]

%% 创建增强数据存储，自动调整大小
augimdsTest = augmentedImageDatastore(inputSize, imdsTest);

%% 循环处理每个输出文件
for f = 1:length(outputFiles)
    fprintf('=== 开始处理文件: %s ===\n', outputFiles{f});
    
    timePerRound = zeros(numRounds,1);
    
    for r = 1:numRounds
        reset(imdsTest);  % 每轮从头读取
        tic;
        classify(netTransfer, augimdsTest);   % 只测推理时间
        timePerRound(r) = toc;
    end
    
    % 保存时间结果
    fid = fopen(outputFiles{f}, 'w');
    if fid == -1
        error('无法打开文件: %s', outputFiles{f});
    end
    for r = 1:numRounds
        fprintf(fid, '%.6f\n', timePerRound(r));
    end
    fclose(fid);
    
    fprintf('预测时间已保存到 %s\n\n', outputFiles{f});
end
