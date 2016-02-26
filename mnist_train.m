clear
%load mnist data
display 'loading data...'
images = single(loadMNISTImages('data/train-images-idx3-ubyte'));
labels = single(loadMNISTLabels('data/train-labels-idx1-ubyte'));