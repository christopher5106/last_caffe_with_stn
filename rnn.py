
import numpy as np

dim = 100
num_batch = 100
mnist_sequence = "mnist_sequence3_sample_8distortions_9x9.npz"
NUM_EPOCH = 300
test_interval = 10

print "Loading data"
data = np.load(mnist_sequence)

x_train, y_train = data['X_train'].reshape((-1, dim, dim)), data['y_train']
x_valid, y_valid = data['X_valid'].reshape((-1, dim, dim)), data['y_valid']
x_test, y_test = data['X_test'].reshape((-1, dim, dim)), data['y_test']

Xt = x_train[:num_batch]
batches_train = x_train.shape[0] // num_batch
batches_valid = x_valid.shape[0] // num_batch

num_steps = y_train.shape[1]

x_train = x_train.reshape(60000, 1, 100, 100)
print "batches_train shape", x_train.shape
print "batches train ", batches_train
print "num steps", num_steps

import sys
sys.path.insert(0, 'python')
import caffe
#caffe.set_mode_gpu()

solver = caffe.RMSPropSolver('rnn_solver.prototxt')
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
for k, v in solver.net.params.items():
    print k
    if len(v): print v[0].data.shape
    if len(v)>1: print v[1].data.shape

niter= NUM_EPOCH * batches_train
train_loss = np.zeros(niter)
#solver.net.params['lstm1'][2].data[256:256*2]=5
solver.net.blobs['clip'].data[...] = np.array([[0,1,1]]*100).reshape(100,3,1)
iter = 0;

for epoch in range(NUM_EPOCH):
    shuffle = np.random.permutation(x_train.shape[0])
    for i in range(batches_train):
        idx = shuffle[i*num_batch:(i+1)*num_batch]
        x_batch = x_train[idx]
        y_batch = y_train[idx]
        #print solver.net.blobs['clip'].data[...]
        #print x_batch.shape
        #print y_batch.shape
        solver.net.blobs['data'].data[...] =x_batch
        solver.net.blobs['label'].data[...]=y_batch
        solver.step(1)
        train_loss[iter] = solver.net.blobs['loss'].data
        if iter % test_interval == 0:
            solver.test_nets[0].blobs['data'].data[...] =x_batch
            solver.test_nets[0].blobs['label'].data[...]=y_batch
            solver.test_nets[0].forward()
            solver.test_nets[0].blobs['accuracy'].data
        print "Iter", iter, train_loss[iter]
        iter+=1

solver.net.save('train_val.caffemodel')
