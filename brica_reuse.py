import numpy as np
from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F
import data

import brica1

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
    def __init__(self, n_input, n_output):
        super(SLPComponent, self).__init__()
        self.model = SLP(n_input, n_output)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

        self.make_in_port("input", n_input)
        self.make_in_port("target", 1)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)
        self.make_out_port("accuracy", 1)

    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)
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

if __name__ == "__main__":
    batchsize = 100
    n_epoch = 1

    mnist = data.load_mnist_data()
    mnist['data'] = mnist['data'].astype(np.float32)
    mnist['data'] /= 255
    mnist['target'] = mnist['target'].astype(np.int32)

    N_train = 60000
    x_train, x_test = np.split(mnist['data'],   [N_train])
    y_train, y_test = np.split(mnist['target'], [N_train])
    N_test = y_test.size

    autoencoder1 = AutoencoderComponent(28**2, 1000)
    autoencoder2 = AutoencoderComponent(1000, 1000)
    autoencoder3 = AutoencoderComponent(1000, 1000)
    slp = SLPComponent(1000, 10)

    brica1.connect((autoencoder1, "output"), (autoencoder2, "input"))
    brica1.connect((autoencoder2, "output"), (autoencoder3, "input"))
    brica1.connect((autoencoder3, "output"), (slp, "input"))

    stacked_autoencoder = brica1.ComponentSet()
    stacked_autoencoder.add_component("autoencoder1", autoencoder1, 1)
    stacked_autoencoder.add_component("autoencoder2", autoencoder2, 2)
    stacked_autoencoder.add_component("autoencoder3", autoencoder3, 3)
    stacked_autoencoder.add_component("slp", slp, 4)

    stacked_autoencoder.make_in_port("input", 28**2)
    stacked_autoencoder.make_in_port("target", 1)
    stacked_autoencoder.make_out_port("output", 1000)
    stacked_autoencoder.make_out_port("loss1", 1)
    stacked_autoencoder.make_out_port("loss2", 1)
    stacked_autoencoder.make_out_port("loss3", 1)
    stacked_autoencoder.make_out_port("loss4", 1)
    stacked_autoencoder.make_out_port("accuracy", 1)

    brica1.alias_in_port((stacked_autoencoder, "input"), (autoencoder1, "input"))
    brica1.alias_out_port((stacked_autoencoder, "output"), (slp, "output"))
    brica1.alias_out_port((stacked_autoencoder, "loss1"), (autoencoder1, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss2"), (autoencoder2, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss3"), (autoencoder3, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss4"), (slp, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "accuracy"), (slp, "accuracy"))
    brica1.alias_in_port((stacked_autoencoder, "target"), (slp, "target"))

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
        sum_loss4 = 0
        sum_accuracy = 0

        for batchnum in xrange(0, N_train, batchsize):
            x_batch = x_train[perm[batchnum:batchnum+batchsize]]
            y_batch = y_train[perm[batchnum:batchnum+batchsize]]

            stacked_autoencoder.get_in_port("input").buffer = x_batch
            stacked_autoencoder.get_in_port("target").buffer = y_batch

            time = agent.step()

            loss1 = stacked_autoencoder.get_out_port("loss1").buffer
            loss2 = stacked_autoencoder.get_out_port("loss2").buffer
            loss3 = stacked_autoencoder.get_out_port("loss3").buffer
            loss4 = stacked_autoencoder.get_out_port("loss4").buffer
            accuracy = stacked_autoencoder.get_out_port("accuracy").buffer

            print "Time: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tAccuracy: {}".format(time, loss1, loss2, loss3, loss4, accuracy)

            sum_loss1 += loss1 * batchsize
            sum_loss2 += loss2 * batchsize
            sum_loss3 += loss3 * batchsize
            sum_loss4 += loss4 * batchsize
            sum_accuracy += accuracy * batchsize

        mean_loss1 = sum_loss1 / N_train
        mean_loss2 = sum_loss2 / N_train
        mean_loss3 = sum_loss3 / N_train
        mean_loss4 = sum_loss3 / N_train
        mean_accuracy = sum_accuracy / N_train

        print "Epoch: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tAccuracy: {}".format(epoch, mean_loss1, mean_loss2, mean_loss3, mean_loss4, mean_accuracy)
        
    threshold = {"mean_loss1" : mean_loss1, "mean_loss2" : mean_loss2, "mean_loss3" : mean_loss3, "mean_loss4" : mean_loss4, "mean_accuracy" : mean_accuracy}
    logger = {"mean_loss1" : 0.0, "mean_loss2" : 0.0, "mean_loss3" : 0.0, "mean_loss4" : 0.0, "mean_accuracy" : 0.0}
    
    
    #reuse except autoencoder2,3,slp
    autoencoder1.last_input_time = 0.0
    autoencoder1.last_output_time = 0.0
    autoencoder4 = AutoencoderComponent(1000, 1000)
    autoencoder5 = AutoencoderComponent(1000, 1000)
    slp2 = SLPComponent(1000, 10)

    brica1.connect((autoencoder1, "output"), (autoencoder4, "input"))
    brica1.connect((autoencoder4, "output"), (autoencoder5, "input"))
    brica1.connect((autoencoder5, "output"), (slp2, "input"))

    stacked_autoencoder2 = brica1.ComponentSet()
    stacked_autoencoder2.add_component("autoencoder1", autoencoder1, 1)
    stacked_autoencoder2.add_component("autoencoder4", autoencoder4, 2)
    stacked_autoencoder2.add_component("autoencoder5", autoencoder5, 3)
    stacked_autoencoder2.add_component("slp2", slp2, 4)

    stacked_autoencoder2.make_in_port("input", 28**2)
    stacked_autoencoder2.make_in_port("target", 1)
    stacked_autoencoder2.make_out_port("output", 1000)
    stacked_autoencoder2.make_out_port("loss1", 1)
    stacked_autoencoder2.make_out_port("loss2", 1)
    stacked_autoencoder2.make_out_port("loss3", 1)
    stacked_autoencoder2.make_out_port("loss4", 1)
    stacked_autoencoder2.make_out_port("accuracy", 1)

    brica1.alias_in_port((stacked_autoencoder2, "input"), (autoencoder1, "input"))
    brica1.alias_out_port((stacked_autoencoder2, "output"), (slp2, "output"))
    brica1.alias_out_port((stacked_autoencoder2, "loss1"), (autoencoder1, "loss"))
    brica1.alias_out_port((stacked_autoencoder2, "loss2"), (autoencoder4, "loss"))
    brica1.alias_out_port((stacked_autoencoder2, "loss3"), (autoencoder5, "loss"))
    brica1.alias_out_port((stacked_autoencoder2, "loss4"), (slp2, "loss"))
    brica1.alias_out_port((stacked_autoencoder2, "accuracy"), (slp2, "accuracy"))
    brica1.alias_in_port((stacked_autoencoder2, "target"), (slp2, "target"))
    
    scheduler = brica1.VirtualTimeSyncScheduler()
    agent2 = brica1.Agent(scheduler)
    module2 = brica1.Module()
    module2.add_component("stacked_autoencoder2", stacked_autoencoder2)
    agent2.add_submodule("module2", module2)
    
    time = 0.0

    for epoch in xrange(n_epoch):
        perm = np.random.permutation(N_train)
        flag_loss1 = True
        flag_loss2 = True
        flag_loss3 = True
        flag_loss4 = True
        flag_accuracy = True
        sum_loss1 = 0
        sum_loss2 = 0
        sum_loss3 = 0
        sum_loss4 = 0
        sum_accuracy = 0

        for batchnum in xrange(0, N_train, batchsize):
            x_batch = x_train[perm[batchnum:batchnum+batchsize]]
            y_batch = y_train[perm[batchnum:batchnum+batchsize]]

            stacked_autoencoder2.get_in_port("input").buffer = x_batch
            stacked_autoencoder2.get_in_port("target").buffer = y_batch
            time = agent2.step()

            loss1 = stacked_autoencoder2.get_out_port("loss1").buffer
            loss2 = stacked_autoencoder2.get_out_port("loss2").buffer
            loss3 = stacked_autoencoder2.get_out_port("loss3").buffer
            loss4 = stacked_autoencoder2.get_out_port("loss4").buffer
            accuracy = stacked_autoencoder2.get_out_port("accuracy").buffer

            print "Time: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tAccuracy: {}".format(time, loss1, loss2, loss3, loss4, accuracy)
            
            if threshold["mean_loss1"] >= loss1 and flag_loss1:
                logger["mean_loss1"] = time - 600.0
                flag_loss1 = False
            if threshold["mean_loss2"] >= loss2 and flag_loss2:
                logger["mean_loss2"] = time - 600.0
                flag_loss2 = False
            if threshold["mean_loss3"] >= loss3 and flag_loss3:
                logger["mean_loss3"] = time - 600.0
                flag_loss3 = False
            if threshold["mean_loss4"] >= loss4 and flag_loss4:
                logger["mean_loss4"] = time - 600.0
                flag_loss4 = False
            if threshold["mean_accuracy"] >= accuracy and flag_accuracy:
                logger["mean_accuracy"] = time - 600.0
                flag_accuracy = False
                

            sum_loss1 += loss1 * batchsize
            sum_loss2 += loss2 * batchsize
            sum_loss3 += loss3 * batchsize
            sum_loss4 += loss4 * batchsize
            sum_accuracy += accuracy * batchsize

        mean_loss1 = sum_loss1 / N_train
        mean_loss2 = sum_loss2 / N_train
        mean_loss3 = sum_loss3 / N_train
        mean_loss4 = sum_loss3 / N_train
        mean_accuracy = sum_accuracy / N_train

        print "Epoch: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tAccuracy: {}".format(epoch, mean_loss1, mean_loss2, mean_loss3, mean_loss4, mean_accuracy)
        print "result:" + str(logger)