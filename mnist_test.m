function [] = mnist_test(net)
    %load mnist data
    display 'loading data...'
    images = single(loadMNISTImages('data/t10k-images-idx3-ubyte'));
    images = reshape(images, 28, 28, 1, []) ;
    labels = single(loadMNISTLabels('data/t10k-labels-idx1-ubyte'));
    labels = labels+1;

    net.layers{end}.class = labels ;
    res = [];
    res = vl_simplenn(net, images, [], res, ...
      'disableDropout', false, ...
      'conserveMemory', false, ...
      'sync', true)
    
end
