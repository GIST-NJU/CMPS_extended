function [followUpTestCase] = shearPlus20degrees(sourceTestCase)
%SHEARPLUS20DEGREES Summary of this function goes here
%   Detailed explanation goes here
    global net2;
    sz = net2.Layers(1).InputSize;
    a = 0.1;
    T = maketform('affine', [1 0 0; a 1 0; 0 0 1] );
    R = makeresampler({'cubic','nearest'},'fill');
    I = imtransform(sourceTestCase,T,R); 
    followUpTestCase = I(1:sz(1),1:sz(2),1:sz(3)); 
end

