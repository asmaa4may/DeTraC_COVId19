function Iout = readAndPreprocessImage(filename)

I = imread(filename);
I=imadjust(I,stretchlim(I),[]);

  % Replicate the image 3 times to create an RGB image.
if ismatrix(I)
   I= cat(3,I,I,I);
end


% Resize the image as required for the CNN.
 Iout= imresize(I, [224 224]); 

end



