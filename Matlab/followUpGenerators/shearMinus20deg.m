function [followUpTestCase] = shearMinus20deg(sourceTestCase)
%https://www.mathworks.com/help/images/padding-and-shearing-an-image-simultaneously.html

    global net2;
    sz = net2.Layers(1).InputSize;
    
    a = 0.1;
    T = maketform('affine', [1 0 0; a 1 0; 0 0 1] );
    R = makeresampler({'cubic','nearest'},'fill');
    I = imtransform(sourceTestCase,T,R); 
    followUpTestCase = I(1:sz(1),1:sz(2),1:sz(3)); 
end

