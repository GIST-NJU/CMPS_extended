function [confArray, failArray] = getRequiredDataFruit360(net, testImgFolder, followUpGenerator)
    global net2;
    net2 = net;
    numOfValidTests = 0;

    % 用 imageDatastore 读取 Fruit360 测试图片
    imdsTest = imageDatastore(testImgFolder, ...
        'IncludeSubfolders', true, ...   % 这里要 true，才能包含子文件夹
        'LabelSource', 'none');          % 不用文件夹名做标签

    numFiles = numel(imdsTest.Files);

    % 预分配
    fileNamesArray = cell(numFiles, 1);
    labelSourceArray = cell(numFiles, 1);
    confidenceSourceArray = cell(numFiles, 1);
    labelFollowUpArray = cell(numFiles, 1);
    confidenceFollowUpArray = cell(numFiles, 1);
    failArray = zeros(numFiles, 1);
    confArray = zeros(numFiles, 1);

    sz = net.Layers(1).InputSize;  % 网络输入尺寸，例如 [224 224 3]

    for k = 1:numFiles
        % 读取图像
        I = readimage(imdsTest, k);

        % resize 到网络输入大小
        Iresized = imresize(I, sz(1:2));

        % 执行源测试用例
        [labelSource, confidenceSource] = classify(net, Iresized);
        conf = sort(confidenceSource, 'descend');

        % === 保留父目录/文件名 ===
        [parentDir, name, ext] = fileparts(imdsTest.Files{k});   % 先分解路径
        [~, parentDirName] = fileparts(parentDir);               % 提取父目录名
        fileNamesArray{k} = fullfile(parentDirName, [name ext]); % 组合 "父目录/文件名"

        labelSourceArray{k} = char(labelSource);
        confidenceSourceArray{k} = confidenceSource;

        % follow-up
        followUpTestCase = followUpGenerator(Iresized);
        [labelFollowUp, confidenceFollowUp] = classify(net, followUpTestCase);

        labelFollowUpArray{k} = char(labelFollowUp);
        confidenceFollowUpArray{k} = confidenceFollowUp;

        numOfValidTests = numOfValidTests + 1;

        % 判断 fail
        if labelSource ~= labelFollowUp
            disp(['fail @ ' fullfile(parentDirName, [name ext])]);  %num2str(k)
            failArray(numOfValidTests) = 1;
        else
            failArray(numOfValidTests) = 0;
        end

        % 保存置信度
        confArray(numOfValidTests) = conf(1);
    end

    % 保存结果
    followUpName = func2str(followUpGenerator);
    save([followUpName '_classification_results.mat'], ...
         'fileNamesArray', 'labelSourceArray', 'confidenceSourceArray', ...
         'labelFollowUpArray', 'confidenceFollowUpArray', 'failArray', 'confArray');
end
