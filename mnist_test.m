function [accuracy] = mnist_test(net)
    setup;
    %load mnist data
    display 'loading data...'
    images = single(loadMNISTImages('data/t10k-images-idx3-ubyte'));
    images = reshape(images, 28, 28, 1, []) ;
    labels = single(loadMNISTLabels('data/t10k-labels-idx1-ubyte'));
    labels = labels;

    net.layers{end} = struct('type', 'softmax') ;
    res = [];
    res = vl_simplenn(net, images, [], res, ...
      'disableDropout', true, ...
      'conserveMemory', false, ...
      'sync', true);
    score=res(end).x;
    
    [~,class] = max(score,[],3);
    class = reshape(class,size(labels));
    class = class-1;
    accuracy = nnz(class==labels)/size(labels,1);
    fprintf('Test set accuracy: %f\n',accuracy);
end
