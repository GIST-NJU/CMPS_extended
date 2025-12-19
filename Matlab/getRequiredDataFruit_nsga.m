function [confArray, failArray] = getRequiredDataFruit_nsga(net, testImgFolder, followUpGenerator)
    global net2;
    net2 = net;
    numOfValidTests = 0;

    % 1. 用 imageDatastore 读取 Fruit360_subset 测试图片
    imdsTest = imageDatastore(testImgFolder, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');  % 使用文件夹名作为标签

    numFiles = numel(imdsTest.Files);

    % 2. 预分配
    confArray = zeros(numFiles, 1);
    failArray = zeros(numFiles, 1);

    % 3. 获取网络输入大小
    sz = net.Layers(1).InputSize;  % e.g. [224 224 3] or [299 299 3]

    % 4. 遍历每张图像
    for k = 1:numFiles
        % 读取图像
        I = readimage(imdsTest, k);

        % resize 到网络输入大小
        I = imresize(I, sz(1:2));

        % 执行 source test
        sourceTestCase = I;
        [labelSource, confidence] = classify(net, sourceTestCase);
        conf = sort(confidence, 'descend');

        % 执行 follow-up test
        followUpTestCase = followUpGenerator(sourceTestCase);
        labelFollowUp = classify(net, followUpTestCase);

        % 计数 + 判断是否 fail
        numOfValidTests = numOfValidTests + 1;
        if labelSource ~= labelFollowUp
            disp(['fail @ ' num2str(k)]);
            failArray(numOfValidTests, 1) = 1;
        else
            failArray(numOfValidTests, 1) = 0;
        end

        % 保存置信度
        confArray(numOfValidTests, 1) = conf(1);
    end
end
