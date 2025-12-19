%% 清空环境
clear; clc; close all;

%% 设置参数
testDir = '/Users/miya_wang/Desktop/CMPS++/ImageNet';  % ImageNet 测试集文件夹
numRounds = 10;

outputFiles = { ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/cmps/imagenet_googlenet.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/cmps++/imagenet_googlenet.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/nsga/imagenet_googlenet.txt', ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/mpss/imagenet_googlenet.txt' ...
};

%% 读取 ImageNet 测试图像
imdsTest = imageDatastore(testDir, ...
    'IncludeSubfolders', false, ...
    'FileExtensions', {'.jpg','.jpeg','.JPEG','.png'});  

numImages = numel(imdsTest.Files);
fprintf('测试集包含 %d 张图片\n', numImages);

% 修复灰度图：统一转为三通道
imdsTest.ReadFcn = @(filename) fix_gray(imread(filename));

%% 加载 MATLAB 自带 GoogLeNet 模型
%net =  resnet50();
net = googlenet;
inputSize = net.Layers(1).InputSize;

%% 循环处理每个输出文件
for f = 1:length(outputFiles)
    fprintf('=== 开始处理文件: %s ===\n', outputFiles{f});
    
    timePerRound = zeros(numRounds,1);
    
    for r = 1:numRounds
        reset(imdsTest);
        augimdsTest = augmentedImageDatastore(inputSize, imdsTest); % 重新创建增强存储
        
        tic;
        classify(net, augimdsTest);  % 推理
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

fprintf('平均推理时间: %.6f 秒\n', mean(timePerRound));


%% 灰度图修复函数
function I = fix_gray(I)
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
end
