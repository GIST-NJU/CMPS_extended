%% 清空环境
clear; clc; close all;

%% 设置参数
testFile = '/Users/miya_wang/Desktop/CMPS++/Fashion-MNIST/t10k-images-idx3-ubyte';  % Fashion-MNIST 测试集文件
numRounds = 10;

outputFiles = { ...
    '/Users/miya_wang/Desktop/results/time/cmps/fashion_lenet5.txt', ...
    '/Users/miya_wang/Desktop/results/time/cmps++/fashion_lenet5.txt', ...
    '/Users/miya_wang/Desktop/results/time/nsga/fashion_lenet5.txt' ...
    '/Users/miya_wang/Desktop/CMPS++/results/time/mpss/fashion_lenet1.txt' ...
};

%% 读取 Fashion-MNIST 测试图像
[testImages, numImages] = read_images(testFile);   % (numImages, 28, 28)
fprintf('测试集包含 %d 张图片\n', numImages);

% 转换为 4D 数组: [H W C N]
testImages = reshape(testImages, numImages, 28, 28);
testImages = permute(testImages, [2,3,1]);       % (28,28,numImages)
testImages = reshape(testImages, 28,28,1,numImages);  % (28,28,1,N)

% 转换为 datastore
imdsTest = arrayDatastore(testImages, 'IterationDimension', 4);

%% 加载保存好的 LeNet1 模型
load('LeNet1_FashionMNIST.mat','net');  % 确认变量名 trainedNet
% load('LeNet5_FashionMNIST.mat','net'); 

%% 获取输入大小
inputSize = net.Layers(1).InputSize;   % 应该是 [28 28 1]

%% 循环处理每个输出文件
for f = 1:length(outputFiles)
    fprintf('=== 开始处理文件: %s ===\n', outputFiles{f});
    
    timePerRound = zeros(numRounds,1);
    
    for r = 1:numRounds
        reset(imdsTest);  % 每轮从头读取整个数据集
        tic;
        classify(net, imdsTest);   % 只测推理时间
        timePerRound(r) = toc;
    end
    
    % 保存每轮时间到 TXT，每行一个时间
    fid = fopen(outputFiles{f}, 'w');
    for r = 1:numRounds
        fprintf(fid, '%.6f\n', timePerRound(r));
    end
    fclose(fid);
    
    fprintf('预测时间已保存到 %s\n\n', outputFiles{f});
end


%% 自定义函数：读取 Fashion-MNIST 图像
function [images, numImages] = read_images(file_path)
    fileID = fopen(file_path, 'rb');
    if fileID == -1
        error('Failed to open image file.');
    end
    
    magic = fread(fileID, 1, 'int32', 'b');    %#ok<NASGU>
    numImages = fread(fileID, 1, 'int32', 'b');
    numRows   = fread(fileID, 1, 'int32', 'b');
    numCols   = fread(fileID, 1, 'int32', 'b');

    % 读取所有像素
    images = fread(fileID, inf, 'unsigned char');
    fclose(fileID);
    
    % 变成 (numImages, numRows, numCols)
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images, [3,2,1]);  % (numImages, numRows, numCols)
end
