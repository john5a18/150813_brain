import numpy as np
from chainer import Variable, FunctionSet, optimizers, cuda
import chainer.functions  as F
import data

class MLP(FunctionSet):
    def __init__(self, n_input, n_output, n_middle):
        super(MLP, self).__init__(
            transform1 = F.Linear(n_input, n_middle),
            transform2 = F.Linear(n_middle, n_output)
        )
    
    def forward(self, x_data, y_data):
        x = Variable(x_data)
        t = Variable(y_data)
        y1 = F.sigmoid(self.transform1(x))
        y2 = F.sigmoid(self.transform2(y1))
        loss = F.softmax_cross_entropy(y2, t)
        accuracy = F.accuracy(y2, t)
        return loss, accuracy

if __name__ == "__main__":
    batchsize = 100
    n_epoch = 20
    
    mnist = data.load_mnist_data()
    mnist['data'] = mnist['data'].astype(np.float32)
    mnist['data'] /= 255
    mnist['target'] = mnist['target'].astype(np.int32)
    
    N_train = 60000
    x_train, x_test = np.split(mnist['data'],   [N_train])
    y_train, y_test = np.split(mnist['target'], [N_train])
    N_test = y_test.size
    
    model = MLP(28**2, 10, 1000)
    optimizer = optimizers.Adam()
    optimizer.setup(model.collect_parameters())
    
    for epoch in xrange(n_epoch):
        perm = np.random.permutation(N_train)
        sum_accuracy = 0
        sum_loss = 0
        
        for batchnum in xrange(0, N_train, batchsize):
            x_batch = x_train[perm[batchnum:batchnum+batchsize]]
            y_batch = y_train[perm[batchnum:batchnum+batchsize]]
            
            optimizer.zero_grads()
            loss, accuracy = model.forward(x_batch, y_batch)
            loss.backward()
            optimizer.update()
            
            sum_accuracy += accuracy.data * batchsize
            sum_loss += loss.data * batchsize
            
        mean_accuracy = sum_accuracy / N_train
        mean_loss = sum_loss / N_train
        print "Train Epoch: {} /Loss: {} /Accuracy {}".format(epoch, mean_loss, mean_accuracy)
        
        
        sum_accuracy = 0
        sum_loss = 0
        
        for batchnum in xrange(0, N_test, batchsize):
            x_batch = x_test[batchnum:batchnum+batchsize]
            y_batch = y_test[batchnum:batchnum+batchsize]
            
            loss, accuracy = model.forward(x_batch, y_batch)
            
            sum_accuracy += accuracy.data * batchsize
            sum_loss += loss.data * batchsize
            
        mean_accuracy = sum_accuracy / N_test
        mean_loss = sum_loss / N_test
        print "Test Epoch: {} /Loss: {} /Accuracy {}".format(epoch, mean_loss, mean_accuracy)