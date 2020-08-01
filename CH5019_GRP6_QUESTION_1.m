clear
clc

%%Number of people
pers = 15; 
%%Number of images per person
no_of_images = 10;
%%Total Number of Images
N = pers*no_of_images;

%%Reading All Images
count = 1;
for i = 1:pers
    for j = 1:no_of_images
        image(:,:,count) = imread(['/Users/Anand/Desktop/Term project 2020/Dataset_Question1/' num2str(i) '/' num2str(j) '.pgm']);
        im_data(:,:,count) = double(image(:,:,count));
        count = count + 1;
    end
end

n = size(im_data,1);
m = size(im_data,2);

%%Mean Shifting 
im_data = im_data - mean(im_data,3);

%%Storing image of each person separately
im_pers = reshape(im_data, n*m, no_of_images, pers);

%%Creating a data matrix to perform SVD on
X = reshape(im_pers, n*m, N);

%%SVD
v_mat = X'*X;

[V,D] = eig(v_mat);

%%Rearranging eigen vectors such that they correspond to eigen values from
%%highest to lowest
for i = 1:size(V,2)
    if i <= size(V,2)/2
        temp(:, i) = V(:, i);
        V(:, i) = V(:, size(V,2) + 1 - i);
        V(:, size(V,2) + 1 - i) = temp(:, i);
    end
end

%%Finding Egien Faces
ef = X*V;
ef = normc(ef);
%%Finding representative image for each face which will be used for
%%identification

for i = 1:pers
    im_rep(:,i) = ef'*mean(im_pers(:,:,i),2);
end

%%Trying to identify images by using our representative matrix
for count = 1:N
    recog(count,1) = ceil(count/no_of_images);
    proj_test = ef'*X(:,count);
    [min_dist, recog(count,2)] = min(sum((proj_test*ones(1,pers) - im_rep).^2, 1));
end

%%Checking Accuracy

correct = sum(recog(:,1) == recog(:,2));

%%Plotting Representative Faces

for i = 1:pers
    subplot(5,3,i), imshow(reshape(mean(im_pers(:,:,i),2),n,m))
    caption = sprintf('Representative Image of Person #%d', i);
    title(caption, 'FontName', 'Times Roman New', 'FontSize', 20);
end


