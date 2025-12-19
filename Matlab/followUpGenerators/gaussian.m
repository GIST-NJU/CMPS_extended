function [followUpTestCase] = gaussian(sourceTestCase)
    followUpTestCase = imgaussfilt(sourceTestCase, 2);
end