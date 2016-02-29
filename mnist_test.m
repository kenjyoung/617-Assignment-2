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
    score = mean(scores,4);
    [~,class] = max(score,[],1);
    class = reshape(class,size(labels));
    class = class-1;
    accuracy = nnz(class==labels)/size(labels,1);
    fprintf('Test set accuracy: %f\n',accuracy);
end
