import numpy as np
from chainer import Variable, FunctionSet, optimizers, cuda
import chainer.functions  as F
import data
import brica1

from logging import getLogger,StreamHandler,DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

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
    
    def encode(self, x_data):
        x = Variable(x_data)
        h = F.sigmoid(self.encoder(x))
        return h.data
        
        
class AutoencoderComponent(brica1.Component):
    def __init__(self, n_input, n_output):
        super(AutoencoderComponent, self).__init__()
        self.model = Autoencoder(n_input, n_output)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())
    
        self.make_in_port("input", n_input)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)
        
    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)
        y_data = self.model.encode(x_data)
        self.results["output"] = y_data
        
        self.optimizer.zero_grads()
        loss = self.model.forward(x_data)
        loss.backward()
        self.optimizer.update()
        self.results["loss"] = loss.data
    
    def encode(self, x_data):
        x = Variable(x_data)
        h = F.sigmoid(self.encoder(x))
        return h.data
    
        
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
        
        
class MLPComponent(brica1.Component):
    def __init__(self, n_input, n_output, n_middle, y_input):
        super(MLPComponent, self).__init__()
        self.model = MLP(n_input, n_output, n_middle)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())
    
        self.make_in_port("input", n_input)
        self.make_in_port("y_input", y_input)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)
        self.make_out_port("accuracy", 1)
        
    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)
        y_data = self.inputs["y_input"].astype(np.float32)
        
        self.optimizer.zero_grads()
        loss, accuracy = self.model.forward(x_data, y_data)
        loss.backward()
        self.optimizer.update()
        self.results["loss"] = loss
        self.results["accuracy"] = accuracy



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
    time = 0.0
    
    component1 = AutoencoderComponent(28**2, 1000)
    component2 = AutoencoderComponent(1000, 1000)
    component3 = AutoencoderComponent(1000, 1000)
    component4 = MLPComponent(1000, 10, 1000, 28*28)
    
    brica1.connect((component1, "output"), (component2, "input"))
    brica1.connect((component2, "output"), (component3, "input"))
    brica1.connect((component3, "output"), (component4, "input"))
    
    stacked_autoencoder = brica1.ComponentSet()
    stacked_autoencoder.add_component("component1", component1, 1)
    stacked_autoencoder.add_component("component2", component2, 2)
    stacked_autoencoder.add_component("component3", component3, 3)
    stacked_autoencoder.add_component("component4", component4, 4)
    stacked_autoencoder.make_in_port("input", 28**2)
    stacked_autoencoder.make_in_port("y_input", 28**2)
    stacked_autoencoder.make_out_port("output", 1000)
    stacked_autoencoder.make_out_port("loss1", 1)
    stacked_autoencoder.make_out_port("loss2", 1)
    stacked_autoencoder.make_out_port("loss3", 1)
    stacked_autoencoder.make_out_port("loss4", 1)
    stacked_autoencoder.make_out_port("accuracy", 1)
    
    brica1.alias_in_port((stacked_autoencoder, "input"),(component1, "input"))
    brica1.alias_out_port((stacked_autoencoder, "output"),(component3, "output"))
    brica1.alias_out_port((stacked_autoencoder, "loss1"),(component1, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss2"),(component2, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss3"),(component3, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss4"),(component4, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "accuracy"),(component4, "accuracy"))
    brica1.alias_in_port((stacked_autoencoder, "y_input"),(component4, "y_input"))
    
    for epoch in xrange(n_epoch):
        perm = np.random.permutation(N_train)
        sum_loss1 = 0
        sum_loss2 = 0
        sum_loss3 = 0
        
        for batchnum in xrange(0, N_train, batchsize):
            x_batch = x_train[perm[batchnum:batchnum+batchsize]]
            y_batch = y_train[perm[batchnum:batchnum+batchsize]]
            
            stacked_autoencoder.get_in_port("input").buffer = x_batch
            stacked_autoencoder.get_in_port("y_input").buffer = y_batch
            logger.debug(x_batch)
            logger.debug(y_batch)
            
            
            stacked_autoencoder.input(time)
            stacked_autoencoder.fire()
            stacked_autoencoder.output(time+1.0)
            
            time += 1.0
            
            loss1 = stacked_autoencoder.get_out_port("loss1").buffer
            loss2 = stacked_autoencoder.get_out_port("loss2").buffer
            loss3 = stacked_autoencoder.get_out_port("loss3").buffer
            
            sum_loss1 += loss1 * batchsize
            sum_loss2 += loss2 * batchsize
            sum_loss3 += loss3 * batchsize
            
        mean_loss1 = sum_loss1 / N_train
        mean_loss2 = sum_loss2 / N_train
        mean_loss3 = sum_loss3 / N_train
        print "Train Epoch: {} /Loss1: {} /Loss2: {} /Loss3: {}".format(epoch, mean_loss1, mean_loss2, mean_loss3)
        
        # sum_loss = 0
        #
        # for batchnum in xrange(0, N_test, batchsize):
        #     x_batch = x_test[batchnum:batchnum+batchsize]
        #
        #     loss = model.forward(x_batch)
        #
        #     sum_loss += loss.data * batchsize
        #
        # mean_loss = sum_loss / N_test
        # print "Test Epoch: {} /Loss: {}".format(epoch, mean_loss)