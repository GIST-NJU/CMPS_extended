function [confArray, failArray] = getRequiredData2(net, testSuiteDirectory, followUpGenerator)
    global net2;
    net2 = net;
    numOfValidTests = 0;
    %testFiles = dir(testSuiteDirectory);
    %disp(testFiles);


    % 读取并解析 t10k-images-idx3-ubyte 文件
    [testImages, ~] = read_images(testSuiteDirectory);  % 使用自定义函数读取图像
    %[testLabels, ~] = read_labels('t10k-labels-idx1-ubyte');  % 读取标签文件

    % 初始化结果数组
    confArray = []; 
    failArray = []; 

    for k = 1:size(testImages, 1)
        I = testImages(k, :, :);  % 获取第k个图像，假设testImages是num_images x 28 x 28的矩阵

        sz = net.Layers(1).InputSize;
        
        % 确保图像大小与网络输入大小匹配
        I = squeeze(I);  % 移除多余的维度

        if size(I, 3) == 1  % 如果图像为灰度图
            % 如果图像尺寸小于网络输入尺寸，进行填充
            if size(I, 1) < sz(1) || size(I, 2) < sz(2)
                padSize = max(sz(1:2) - size(I, 1:2), 0);
                I = padarray(I, padSize, 'post');
            end
            
            % 调整图像大小
            I = I(1:sz(1), 1:sz(2), 1:sz(3));
            sourceTestCase = I(1:sz(1), 1:sz(2), 1:sz(3));
            
            % 执行源测试用例
            [labelSource, confidence] = classify(net, sourceTestCase);
            conf = sort(confidence, 'descend');
            
            % 执行后续测试用例
            followUpTestCase = followUpGenerator(sourceTestCase);
            labelFollowUp = classify(net, followUpTestCase);
            numOfValidTests = numOfValidTests + 1;
            
            % 判断是否失败
            if labelSource ~= labelFollowUp
                disp('fail');
                failArray(numOfValidTests, 1) = 1;
            else
                failArray(numOfValidTests, 1) = 0;
            end
            confArray(numOfValidTests, 1) = conf(1);
        end
    end
end

% 读取图像的自定义函数
function [images, numImages] = read_images(file_path)
    % 打开文件并读取数据
    fileID = fopen(file_path, 'rb');
    if fileID == -1
        error('Failed to open image file.');
    end
    
    % 读取文件头部
    magic = fread(fileID, 1, 'int32', 'b');  % 魔数
    numImages = fread(fileID, 1, 'int32', 'b');  % 图像数量
    numRows = fread(fileID, 1, 'int32', 'b');  % 行数
    numCols = fread(fileID, 1, 'int32', 'b');  % 列数

    % 读取图像数据
    images = fread(fileID, inf, 'unsigned char');
    fclose(fileID);
    
    % 将数据重塑为正确的形状 (numImages, numRows, numCols)
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images, [3, 2, 1]);  % 转换维度为 (numImages, numRows, numCols)
end
