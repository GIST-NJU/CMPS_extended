%% This script gets all the required experimental data for executing the experiments;
% After running it, the workspace needs to be stored.

clear;
clc;
rng(0); % Initial seed to allow reproducibility of results
addpath('followUpGenerators');
modelfile = 'Fruit360_MobileNetV2.mat'
loadedData = load(modelfile)
net = loadedData.trainedNet; %googlenet; %resnet50() %resnet101;
directory = '/Users/miya_wang/Desktop/Fruit360/fruit360_subset';
global confArr
[confArr, failArrFlipLeftRight] = getRequiredDataFruit_nsga(net,directory,@flipLeftRight);
[~, failArrgaussian] = getRequiredDataFruit_nsga(net,directory,@gaussian);
[~, failArrRotatePlus] = getRequiredDataFruit_nsga(net,directory,@rotatePlus30deg);
[~, failArrcolored] = getRequiredDataFruit_nsga(net,directory,@colored);
[~, failArrbrightness] = getRequiredDataFruit_nsga(net,directory,@brightness);