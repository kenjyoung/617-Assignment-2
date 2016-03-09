function [accuracy] = mnist_test(nets)
    setup;
    %load mnist data
    display 'loading data...'
    images = single(loadMNISTImages('data/t10k-images-idx3-ubyte'));
    images = reshape(images, 28, 28, 1, []) ;
    labels = single(loadMNISTLabels('data/t10k-labels-idx1-ubyte'));
    labels = labels;
    for i=1:size(nets,2)
        net = nets(i);
        net.layers{end} = struct('type', 'softmax') ;
        res = [];
        res = vl_simplenn(net, images, [], res, ...
          'disableDropout', true, ...
          'conserveMemory', false, ...
          'sync', true);
        scores(:,:,:,i)=res(end).x;
    end
    [~,class] = max(scores,[],1);
    for i=1:size(nets,2)
        net_class = class(:,:,:,i);
        net_class = reshape(net_class,size(labels));
        net_class = net_class-1;
        net_accuracy(i) = nnz(net_class==labels)/size(labels,1);
    end
    fprintf('Individual accuracy: (%.4f,%.4f,%.4f,%.4f,%.4f)\n', net_accuracy);
    class = mode(class,4);
    class = reshape(class,size(labels));
    class = class-1;
    nnz(class~=labels)
    accuracy = nnz(class==labels)/size(labels,1);
    
    figure;
    wrong_images = images(:,:,class~=labels);
    given_labels = labels(class~=labels);
    true_class = class(class~=labels);
    rows = ceil(sqrt(size(wrong_images,3)));
    for i=1:size(wrong_images,3)
        image = wrong_images(:,:,i)/255;
        I = subplot(rows,rows, i);
        subimage(image);
        %set(I,'Position',[mod(i,rows)/rows,floor(i/rows)/rows,0.9/rows,0.9/rows])
        axis off
        title(sprintf('l:%d, c:%d',given_labels(i),true_class(i)))
    end
    
    
    
    fprintf('Overall test set accuracy: %.4f\n',accuracy);
    
end
