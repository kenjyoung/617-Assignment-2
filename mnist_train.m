function [net, info] = mnist_train()
    %load mnist data
    display 'loading data...'
    images = single(loadMNISTImages('data/train-images-idx3-ubyte'));
    labels = single(loadMNISTLabels('data/train-labels-idx1-ubyte'))';
    image_data.data = images;
    image_data.id = 1:size(images,3);
    image_data.labels = labels+1;
    image_data.set = ones(size(labels));
    image_data.set(1:end/10) = 2;
    
    
    imdb.images = image_data;
    
    trainOpts.batchSize = 10 ;
    trainOpts.numEpochs = 15 ;
    trainOpts.continue = true ;
    trainOpts.useGpu = false ;
    trainOpts.learningRate = 0.03 ;
    trainOpts.expDir = 'data/mnist-experiment' ;

    net = initializeNetwork();
    [net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;
    save net
end

function [images, labels] = getBatch(imdb, batch)
    images = imdb.images.data(:,:,batch);
    images = reshape(images, 28, 28, 1, []) ;
    labels = imdb.images.labels(1,batch);
end