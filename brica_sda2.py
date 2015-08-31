import numpy as np
from chainer import Variable, FunctionSet, optimizers, cuda
import chainer.functions  as F
import data
import brica1

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
        
    # new
    def predict(self, x_data):
        x = Variable(x_data)
        y1 = F.sigmoid(self.transform1(x))
        y2 = F.sigmoid(self.transform2(y1))
        return y2.data
        
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
        
        
class MLPComponent(brica1.Component):
    def __init__(self, n_input, n_output, n_middle):
        super(MLPComponent, self).__init__()
        self.model = MLP(n_input, n_output, n_middle)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())
    
        self.make_in_port("input", n_input)
        self.make_in_port("target", 1) # new
        # self.make_in_port("y_input", y_input)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)
        self.make_out_port("accuracy", 1)
        
    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)
        # y_data = self.inputs["y_input"].astype(np.float32)
        t_data = self.inputs["target"].astype(np.int32)
        
        self.optimizer.zero_grads()
        loss, accuracy = self.model.forward(x_data, t_data)
        loss.backward()
        self.optimizer.update()
        self.results["loss"] = loss.data
        self.results["accuracy"] = accuracy.data
        
        y_data = self.model.predict(x_data)
        self.results["output"] = y_data
        
        
        
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
        
        self.optimizer.zero_grads()
        loss = self.model.forward(x_data)
        loss.backward()
        self.optimizer.update()
        self.results["loss"] = loss.data

        y_data = self.model.encode(x_data)
        self.results["output"] = y_data
    
    # def encode(self, x_data):
    #     x = Variable(x_data)
    #     h = F.sigmoid(self.encoder(x))
    #     return h.data



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
    mlp = MLPComponent(1000, 10, 1000)
    
    brica1.connect((component1, "output"), (component2, "input"))
    brica1.connect((component2, "output"), (component3, "input"))
    brica1.connect((component3, "output"), (mlp, "input"))
    
    stacked_autoencoder = brica1.ComponentSet()
    stacked_autoencoder.add_component("component1", component1, 1)
    stacked_autoencoder.add_component("component2", component2, 2)
    stacked_autoencoder.add_component("component3", component3, 3)
    stacked_autoencoder.add_component("mlp", mlp, 4)
    
    stacked_autoencoder.make_in_port("input", 28**2)
    stacked_autoencoder.make_in_port("target", 1)
    stacked_autoencoder.make_out_port("output", 1000)
    stacked_autoencoder.make_out_port("loss1", 1)
    stacked_autoencoder.make_out_port("loss2", 1)
    stacked_autoencoder.make_out_port("loss3", 1)
    stacked_autoencoder.make_out_port("loss4", 1)
    stacked_autoencoder.make_out_port("accuracy", 1)
    
    brica1.alias_in_port((stacked_autoencoder, "input"),(component1, "input"))
    brica1.alias_out_port((stacked_autoencoder, "output"),(mlp, "output")) #here!!
    brica1.alias_out_port((stacked_autoencoder, "loss1"),(component1, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss2"),(component2, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss3"),(component3, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss4"),(mlp, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "accuracy"),(mlp, "accuracy"))
    brica1.alias_in_port((stacked_autoencoder, "target"),(mlp, "target"))
    
    scheduler = brica1.VirtualTimeSyncScheduler()
    agent = brica1.Agent(scheduler)
    module = brica1.Module()
    module.add_component("stacked_autoencoder", stacked_autoencoder)
    agent.add_submodule("module", module)
    
    time = 0.0
    
    for epoch in xrange(n_epoch):
        perm = np.random.permutation(N_train)
        sum_loss1 = 0
        sum_loss2 = 0
        sum_loss3 = 0
        
        # new
        sum_loss4 = 0
        sum_accuracy = 0
        
        for batchnum in xrange(0, N_train, batchsize):
            x_batch = x_train[perm[batchnum:batchnum+batchsize]]
            y_batch = y_train[perm[batchnum:batchnum+batchsize]]
            
            stacked_autoencoder.get_in_port("input").buffer = x_batch
            stacked_autoencoder.get_in_port("target").buffer = y_batch
            
            time = agent.step()
            
            
            # stacked_autoencoder.input(time)
            # stacked_autoencoder.fire()
            # stacked_autoencoder.output(time+1.0)
            #
            # time += 1.0
            
            loss1 = stacked_autoencoder.get_out_port("loss1").buffer
            loss2 = stacked_autoencoder.get_out_port("loss2").buffer
            loss3 = stacked_autoencoder.get_out_port("loss3").buffer
            
            # new
            loss4 = stacked_autoencoder.get_out_port("loss4").buffer
            accuracy = stacked_autoencoder.get_out_port("accuracy").buffer

            print "Time: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tAccuracy: {}".format(time, loss1, loss2, loss3, loss4, accuracy)
            
            sum_loss1 += loss1 * batchsize
            sum_loss2 += loss2 * batchsize
            sum_loss3 += loss3 * batchsize
            sum_loss4 += loss4 * batchsize
            sum_accuracy += sum_accuracy * batchsize
            
        mean_loss1 = sum_loss1 / N_train
        mean_loss2 = sum_loss2 / N_train
        mean_loss3 = sum_loss3 / N_train
        mean_loss4 = sum_loss4 / N_train
        mean_accuracy = sum_accuracy / N_train
        # print "Train Epoch: {} /Loss1: {} /Loss2: {} /Loss3: {}".format(epoch, mean_loss1, mean_loss2, mean_loss3)
        print "Epoch: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tAccuracy: {}".format(epoch, mean_loss1, mean_loss2, mean_loss3, mean_loss4, mean_accuracy)
