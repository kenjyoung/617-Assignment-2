clear
%load mnist data
display 'loading data...'
images = single(loadMNISTImages('data/t10k-images-idx3-ubyte'));
labels = single(loadMNISTLabels('data/t10k-labels-idx1-ubyte'));