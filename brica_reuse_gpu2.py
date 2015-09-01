#coding: utf-8
import argparse
import numpy as np
from chainer import Variable, FunctionSet, optimizers, cuda
import chainer.functions  as F
import data

import brica1

'''
reference
https://github.com/wbap/V1/blob/master/python/examples/chainer_sda.py
'''

class SLP(FunctionSet):
    def __init__(self, n_input, n_output):
        super(SLP, self).__init__(
            transform=F.Linear(n_input, n_output)
        )

    def forward(self, x_data, y_data):
        x = Variable(x_data)
        t = Variable(y_data)
        y = F.sigmoid(self.transform(x))
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy

    def predict(self, x_data):
        x = Variable(x_data)
        y = F.sigmoid(self.transform(x))
        return y.data

class Autoencoder(FunctionSet):
    def __init__(self, n_input, n_output):
        super(Autoencoder, self).__init__(
            encoder=F.Linear(n_input, n_output),
            decoder=F.Linear(n_output, n_input)
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

class SLPComponent(brica1.Component):
    def __init__(self, n_input, n_output, use_gpu=False):
        super(SLPComponent, self).__init__()
        self.model = SLP(n_input, n_output)
        self.optimizer = optimizers.Adam()
        
        self.make_in_port("input", n_input)
        self.make_in_port("target", 1)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)
        self.make_out_port("accuracy", 1)
        
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.model.to_gpu()

        self.optimizer.setup(self.model)
        # self.optimizer.setup(self.model.collect_parameters())


    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)
        t_data = self.inputs["target"].astype(np.int32)
        
        if self.use_gpu:
            x_data = cuda.to_gpu(x_data)
            t_data = cuda.to_gpu(t_data)

        self.optimizer.zero_grads()
        loss, accuracy = self.model.forward(x_data, t_data)
        loss.backward()
        self.optimizer.update()

        y_data = self.model.predict(x_data)
        
        self.results["loss"] = cuda.to_cpu(loss.data)
        self.results["accuracy"] = cuda.to_cpu(accuracy.data)
        self.results["output"] = cuda.to_cpu(y_data)

class AutoencoderComponent(brica1.Component):
    def __init__(self, n_input, n_output, use_gpu=False):
        super(AutoencoderComponent, self).__init__()
        self.model = Autoencoder(n_input, n_output)
        self.optimizer = optimizers.Adam()
        
        self.make_in_port("input", n_input)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)
        
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.model.to_gpu()
        
        self.optimizer.setup(self.model)
        # self.optimizer.setup(self.model.collect_parameters())
        
    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)

        if self.use_gpu:
            x_data = cuda.to_gpu(x_data)
        
        self.optimizer.zero_grads()
        loss = self.model.forward(x_data)
        loss.backward()
        self.optimizer.update()

        y_data = self.model.encode(x_data)
        
        self.results["loss"] = cuda.to_cpu(loss.data)
        self.results["output"] = cuda.to_cpu(y_data)


class MySetupper:
    def __init__(self, autoencoder1, autoencoder2, autoencoder3, slp):
        self.autoencoder1 = autoencoder1
        self.autoencoder2 = autoencoder2
        self.autoencoder3 = autoencoder3
        self.slp = slp
        self.stacked_autoencoder = brica1.ComponentSet()

        self.scheduler = brica1.VirtualTimeSyncScheduler()
        self.agent = brica1.Agent(self.scheduler)
        self.module = brica1.Module()
        
        
    def connection_setup(self):
        brica1.connect((self.autoencoder1, "output"), (self.autoencoder2, "input"))
        brica1.connect((self.autoencoder2, "output"), (self.autoencoder3, "input"))
        brica1.connect((self.autoencoder3, "output"), (self.slp, "input"))

        self.stacked_autoencoder.add_component("autoencoder1", self.autoencoder1, 1)
        self.stacked_autoencoder.add_component("autoencoder2", self.autoencoder2, 2)
        self.stacked_autoencoder.add_component("autoencoder3", self.autoencoder3, 3)
        self.stacked_autoencoder.add_component("slp", self.slp, 4)

        self.stacked_autoencoder.make_in_port("input", 28**2)
        self.stacked_autoencoder.make_in_port("target", 1)
        self.stacked_autoencoder.make_out_port("output", 1000)
        self.stacked_autoencoder.make_out_port("loss1", 1)
        self.stacked_autoencoder.make_out_port("loss2", 1)
        self.stacked_autoencoder.make_out_port("loss3", 1)
        self.stacked_autoencoder.make_out_port("loss4", 1)
        self.stacked_autoencoder.make_out_port("accuracy", 1)

        brica1.alias_in_port((self.stacked_autoencoder, "input"), (self.autoencoder1, "input"))
        brica1.alias_out_port((self.stacked_autoencoder, "output"), (self.slp, "output"))
        brica1.alias_out_port((self.stacked_autoencoder, "loss1"), (self.autoencoder1, "loss"))
        brica1.alias_out_port((self.stacked_autoencoder, "loss2"), (self.autoencoder2, "loss"))
        brica1.alias_out_port((self.stacked_autoencoder, "loss3"), (self.autoencoder3, "loss"))
        brica1.alias_out_port((self.stacked_autoencoder, "loss4"), (self.slp, "loss"))
        brica1.alias_out_port((self.stacked_autoencoder, "accuracy"), (self.slp, "accuracy"))
        brica1.alias_in_port((self.stacked_autoencoder, "target"), (self.slp, "target"))
        return True
    
    def scheduler_setup(self):
        self.module.add_component("stacked_autoencoder", self.stacked_autoencoder)
        self.agent.add_submodule("module", self.module)
        return True
        
    def setup(self):
        self.connection_setup()
        self.scheduler_setup()
        return True
        
    def run(self, n_epoch, N_train, batchsize, x_train, y_train, threshold=None):
        if threshold:
            self.logger = {"mean_loss1" : 0.0, "mean_loss2" : 0.0, "mean_loss3" : 0.0, "mean_loss4" : 0.0, "mean_accuracy" : 0.0}
            self.flag_loss1 = True
            self.flag_loss2 = True
            self.flag_loss3 = True
            self.flag_loss4 = True
            self.flag_accuracy = True

        f = open("log.txt", "a")
        for epoch in xrange(n_epoch):
            perm = np.random.permutation(N_train)
            self.sum_loss1 = 0
            self.sum_loss2 = 0
            self.sum_loss3 = 0
            self.sum_loss4 = 0
            self.sum_accuracy = 0

            for batchnum in xrange(0, N_train, batchsize):
                x_batch = x_train[perm[batchnum:batchnum+batchsize]]
                y_batch = y_train[perm[batchnum:batchnum+batchsize]]

                self.stacked_autoencoder.get_in_port("input").buffer = x_batch
                self.stacked_autoencoder.get_in_port("target").buffer = y_batch

                time = self.agent.step()

                self.loss1 = self.stacked_autoencoder.get_out_port("loss1").buffer
                self.loss2 = self.stacked_autoencoder.get_out_port("loss2").buffer
                self.loss3 = self.stacked_autoencoder.get_out_port("loss3").buffer
                self.loss4 = self.stacked_autoencoder.get_out_port("loss4").buffer
                self.accuracy = self.stacked_autoencoder.get_out_port("accuracy").buffer
                
                # log = "Time: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tAccuracy: {}\n".format(time, self.loss1, self.loss2, self.loss3, self.loss4, self.accuracy)
#                 print log
#                 f.write(log)
                
                self.sum_loss1 += self.loss1 * batchsize
                self.sum_loss2 += self.loss2 * batchsize
                self.sum_loss3 += self.loss3 * batchsize
                self.sum_loss4 += self.loss4 * batchsize
                self.sum_accuracy += self.accuracy * batchsize

            self.mean_loss1 = self.sum_loss1 / N_train
            self.mean_loss2 = self.sum_loss2 / N_train
            self.mean_loss3 = self.sum_loss3 / N_train
            self.mean_loss4 = self.sum_loss4 / N_train
            self.mean_accuracy = self.sum_accuracy / N_train
            
            if threshold:
                if threshold["mean_loss1"] >= self.loss1 and self.flag_loss1:
                    self.logger["mean_loss1"] = epoch
                    self.flag_loss1 = False
                if threshold["mean_loss2"] >= self.loss2 and self.flag_loss2:
                    self.logger["mean_loss2"] = epoch
                    self.flag_loss2 = False
                if threshold["mean_loss3"] >= self.loss3 and self.flag_loss3:
                    self.logger["mean_loss3"] = epoch
                    self.flag_loss3 = False
                if threshold["mean_loss4"] >= self.loss4 and self.flag_loss4:
                    self.logger["mean_loss4"] = epoch
                    self.flag_loss4 = False
                if threshold["mean_accuracy"] <= self.accuracy and self.flag_accuracy:
                    self.logger["mean_accuracy"] = epoch
                    self.flag_accuracy = False

            log = "Epoch: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tAccuracy: {}\n".format(epoch, self.mean_loss1, self.mean_loss2, self.mean_loss3, self.mean_loss4, self.mean_accuracy)
            f.write(log)
            print log
        f.close()
        
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chainer-BriCa integration")
    parser.add_argument("--gpu", "-g", default=-1, type=int, help="GPU ID")
    
    args = parser.parse_args()
    
    use_gpu=False
    if args.gpu >= 0:
        use_gpu = True
        cuda.get_device(args.gpu).use()
    
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

    #setup
    autoencoder1 = AutoencoderComponent(28**2, 1000, use_gpu=use_gpu)
    autoencoder2 = AutoencoderComponent(1000, 1000, use_gpu=use_gpu)
    autoencoder3 = AutoencoderComponent(1000, 1000, use_gpu=use_gpu)
    slp = SLPComponent(1000, 10)
    
    #setup
    original = MySetupper(autoencoder1, autoencoder2, autoencoder3, slp)
    original.setup()
    
    time = 0.0
    
    original.run(n_epoch, N_train, batchsize, x_train, y_train)
    threshold = {"mean_loss1" : original.mean_loss1, "mean_loss2" : original.mean_loss2, "mean_loss3" : original.mean_loss3, "mean_loss4" : original.mean_loss4, "mean_accuracy" : original.mean_accuracy}

    f = open("log.txt", "a")
    result = "threshold:" + str(threshold) + "\n"
    f.write(result)
    f.close()
    print "threshold:" + str(threshold)
    
    #reuse except autoencoder2,3,slp
    #reset autoencoder1
    autoencoder1.last_input_time = 0.0
    autoencoder1.last_output_time = 0.0
    autoencoder4 = AutoencoderComponent(1000, 1000, use_gpu=use_gpu)
    autoencoder5 = AutoencoderComponent(1000, 1000, use_gpu=use_gpu)
    slp2 = SLPComponent(1000, 10)
    
    original2 = MySetupper(autoencoder1, autoencoder4, autoencoder5, slp2)
    original2.setup()
    
    time = 0.0
    
    original2.run(n_epoch, N_train, batchsize, x_train, y_train, threshold)
    f = open("log.txt", "a")
    result = "result:" + str(original2.logger) + "\n"
    f.write(result)
    f.close()
    print "result:" + str(original2.logger)
    
    
    #reuse except autoencoder1,3,slp
    #reset autoencoder2
    autoencoder6 = AutoencoderComponent(28**2, 1000, use_gpu=use_gpu)
    autoencoder2.last_input_time = 0.0
    autoencoder2.last_output_time = 0.0
    autoencoder7 = AutoencoderComponent(1000, 1000, use_gpu=use_gpu)
    slp3 = SLPComponent(1000, 10)
    
    original3 = MySetupper(autoencoder6, autoencoder2, autoencoder7, slp3)
    original3.setup()
    
    time = 0.0
    
    original3.run(n_epoch, N_train, batchsize, x_train, y_train, threshold)
    f = open("log.txt", "a")
    result = "result:" + str(original3.logger) + "\n"
    f.write(result)
    f.close()
    print "result:" + str(original3.logger)
    

    #reuse except autoencoder1,2,slp
    #reset autoencoder2
    autoencoder8 = AutoencoderComponent(28**2, 1000, use_gpu=use_gpu)
    autoencoder9 = AutoencoderComponent(1000, 1000, use_gpu=use_gpu)
    autoencoder3.last_input_time = 0.0
    autoencoder3.last_output_time = 0.0
    slp4 = SLPComponent(1000, 10)
    
    original4 = MySetupper(autoencoder8, autoencoder9, autoencoder3, slp4)
    original4.setup()
    
    time = 0.0
    
    original4.run(n_epoch, N_train, batchsize, x_train, y_train, threshold)
    f = open("log.txt", "a")
    result = "result:" + str(original4.logger) + "\n"
    f.write(result)
    f.close()
    print "result:" + str(original4.logger)
    

    #reuse except autoencoder1,2,3
    #reset autoencoder2
    autoencoder10 = AutoencoderComponent(28**2, 1000, use_gpu=use_gpu)
    autoencoder11 = AutoencoderComponent(1000, 1000, use_gpu=use_gpu)
    autoencoder12 = AutoencoderComponent(1000, 1000, use_gpu=use_gpu)
    slp.last_input_time = 0.0
    slp.last_output_time = 0.0
    
    original5 = MySetupper(autoencoder10, autoencoder11, autoencoder12, slp)
    original5.setup()
    
    time = 0.0
    
    original5.run(n_epoch, N_train, batchsize, x_train, y_train, threshold)
    f = open("log.txt", "a")
    result = "result:" + str(original5.logger) + "\n"
    f.write(result)
    f.close()
    print "result:" + str(original5.logger)