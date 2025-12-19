function [followUpTestCase] = brightness(sourceTestCase)
    followUpTestCase = imadd(sourceTestCase, 1.3 * 255);
end