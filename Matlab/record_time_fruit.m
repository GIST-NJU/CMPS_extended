%% 清空环境
clear; clc; close all;

%% 设置参数
testDir = '/Users/miya_wang/Desktop/CMPS++/Fruit360/fruit360_subset';  % Fruit360 图像目录
numRounds = 10;

outputFiles = { ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/cmps/fruit360_mobilenetv2.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/cmps++/fruit360_mobilenetv2.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/nsga/fruit360_mobilenetv2.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/mpss/fruit360_mobilenetv2.txt' ...
};

%% 读取 Fruit360 测试图像
imdsTest = imageDatastore(testDir, ...
    'IncludeSubfolders', true, ...     % 递归读取子文件夹
    'LabelSource', 'foldernames', ...  % 文件夹名作为标签
    'FileExtensions', {'.jpg', '.jpeg', '.JPG', '.JPEG'});

numImages = numel(imdsTest.Files);
fprintf('测试集包含 %d 张图片\n', numImages);

%% 加载训练好的 ShuffleNet 模型
load('Fruit360_MobileNetV2.mat', 'trainedNet');  % 模型变量名为 transfer
%load('Fruit360_ShuffleNet.mat', 'trainedNet'); 
net = trainedNet;

%% 检查模型输入尺寸
inputSize = net.Layers(1).InputSize;
fprintf('模型输入尺寸: [%d %d %d]\n', inputSize);

%% 创建增强数据存储（自动调整图像大小）
augimdsTest = augmentedImageDatastore(inputSize, imdsTest);

%% 循环多轮测试推理时间
for f = 1:length(outputFiles)
    fprintf('=== 开始处理文件: %s ===\n', outputFiles{f});
    
    timePerRound = zeros(numRounds,1);
    
    for r = 1:numRounds
        reset(imdsTest);  % 每轮从头开始读取
        
        tic;
        classify(net, augimdsTest);   % 测试推理时间
        timePerRound(r) = toc;
    end
    
    % 保存每轮时间结果
    fid = fopen(outputFiles{f}, 'w');
    if fid == -1
        error('无法打开文件: %s', outputFiles{f});
    end
    for r = 1:numRounds
        fprintf(fid, '%.6f\n', timePerRound(r));
    end
    fclose(fid);
    
    fprintf('预测时间已保存到 %s\n', outputFiles{f});
    fprintf('平均推理时间: %.6f 秒\n\n', mean(timePerRound));
end
