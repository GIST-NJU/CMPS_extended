function [followUpTestCase] = rotateMinus30deg(sourceTestCase)
    followUpTestCase = imrotate(sourceTestCase,-5,'bilinear','crop');
end

