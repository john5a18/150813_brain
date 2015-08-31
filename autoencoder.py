import numpy as np
from chainer import Variable, FunctionSet, optimizers, cuda
import chainer.functions  as F
import data

class Autoencoder(FunctionSet):
    def __init__(self, n_input, n_output):
        super(Autoencoder, self).__init__(
            encoder = F.Linear(n_input, n_output),
            decoder = F.Linear(n_output, n_input)
        )
    
    def forward(self, x_data):
        x = Variable(x_data)
        t = Variable(x_data)
        x = F.dropout(x)
        h = F.sigmoid(self.encoder(x))
        y = F.sigmoid(self.decoder(h))
        loss = F.mean_squared_error(y, t)
        return loss

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
    
    model = Autoencoder(28**2, 1000)
    optimizer = optimizers.Adam()
    optimizer.setup(model.collect_parameters())
    
    for epoch in xrange(n_epoch):
        perm = np.random.permutation(N_train)
        sum_loss = 0
        
        for batchnum in xrange(0, N_train, batchsize):
            x_batch = x_train[perm[batchnum:batchnum+batchsize]]
            
            optimizer.zero_grads()
            loss = model.forward(x_batch)
            loss.backward()
            optimizer.update()
            
            sum_loss += loss.data * batchsize
            
        mean_loss = sum_loss / N_train
        print "Train Epoch: {} /Loss: {}".format(epoch, mean_loss
        
        
        sum_accuracy = 0
        sum_loss = 0
        
        for batchnum in xrange(0, N_test, batchsize):
            x_batch = x_test[batchnum:batchnum+batchsize]
            
            loss = model.forward(x_batch)
            
            sum_loss += loss.data * batchsize
            
        mean_loss = sum_loss / N_test
        print "Test Epoch: {} /Loss: {}".format(epoch, mean_loss)