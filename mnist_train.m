function [nets, info] = mnist_train()
    setup;
    %load mnist data
    display 'loading data...';
    images = single(loadMNISTImages('data/train-images-idx3-ubyte'));
    im_width = size(images,2);
    im_height = size(images,1);
    labels = single(loadMNISTLabels('data/train-labels-idx1-ubyte'))';
    labels = labels+1;
    %augment data set with translated data
    for i=1:size(images,3)
        new_labels(1,4*(i-1)+1:4*(i-1)+4) = labels(1,i);
        new_images(:,:,4*(i-1)+1) = [zeros(1,im_width);images(2:end,:,i)];
        new_images(:,:,4*(i-1)+2) = [images(1:end-1,:,i);zeros(1,im_width)];
        new_images(:,:,4*(i-1)+3) = [zeros(im_height,1), images(:,2:end,i)];
        new_images(:,:,4*(i-1)+4) = [images(:,1:end-1,i), zeros(im_height,1)];
    end
    
    images = cat(3,images,new_images);
    labels = [labels,new_labels];
    shuffle = randperm(size(images,3));
    %shuffle data
    images = images(:,:,shuffle);
    labels = labels(shuffle);
    image_data.data = images;
    image_data.id = 1:size(images,3);
    image_data.labels = labels;
    image_data.set = ones(size(labels));
    image_data.set(1:floor(size(images,3)/20)) = 2;
    
    lambda = 0.1 ;
    %this eta differs from the one given by the source network
    %the eta they provided was found to not work as well
    eta = 0.03 ;
    num_nets = 5 ;
    
    imdb.images = image_data;
    trainOpts.batchSize = 10 ;
    trainOpts.numEpochs = 40 ;
    trainOpts.continue = false ;
    trainOpts.useGpu = false ;
    trainOpts.learningRate = eta;
    %lambda multiplied to agree with definition used in source network
    trainOpts.weightDecay = lambda*(trainOpts.batchSize/size(images,3));
    %source network does not use momentum
    trainOpts.momentum = 0.0 ;
    trainOpts.expDir = 'data/mnist-experiment' ;
    
    for i=1:num_nets
        net = initializeNetwork();
        [net,~] = cnn_train(net, imdb, @getBatch, trainOpts);
        nets(i) = net;
        save nets
    end
end

function [images, labels] = getBatch(imdb, batch)
    images = imdb.images.data(:,:,batch);
    images = reshape(images, 28, 28, 1, []) ;
    labels = imdb.images.labels(1,batch);
end