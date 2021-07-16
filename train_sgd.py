from ml_utils import *
import copy
from torch import optim
from torch.nn.utils import clip_grad_norm_
import time
from utils import *
import wandb


class plainSGD:
    def __init__(self, device, net, optimizer, criterion, params):
        self.TopLevel = 0
        self.num_levels = 1

        self.train_acc = [float(0.0) for i in range(self.num_levels)]
        self.train_total = [int(0.0) for i in range(self.num_levels)]
        self.train_correct = [int(0.0) for i in range(self.num_levels)]
        self.train_loss = [float(0.0) for i in range(self.num_levels)]

        self.loss_corr_cnt = [int(1) for i in range(self.num_levels)]
        self.loss_good_corr_cnt = [int(0) for i in range(self.num_levels)]
        self.loss_good_corr_ratio = [float(0.0) for i in range(self.num_levels)]

        self.nets = [net]
        self.criteria = [criterion]
        self.optimizers = [optimizer]
        self.device = device

        self.params = params
        self.loss_out = 0.0
        self.work = 0.0

        self.cycle_num = 0

        # self.forward_cost = 0.4
        # self.loss_cost = 0.005
        # self.step_cost = 0.005
        # self.backprop_cost = 0.5

        self.forward_cost = 0.0
        self.loss_cost = 0.0
        self.step_cost = 0.0
        self.backprop_cost = 1.0

        self.transfer_cost = 0.01
        self.trainloader = []

        self.sample_rate = int(params.get("sample_rate"))
        self.val_acc = [float(0.0) for i in range(params.get("num_levels"))]
        # self.val_outputs = [float(0.0) for i in range(params.get("num_levels"))]
        self.val_correct = [float(0.0) for i in range(params.get("num_levels"))]
        self.val_loss = [float(0.0) for i in range(params.get("num_levels"))]
        self.val_num = [0 for i in range(params.get("num_levels"))]
        self.val_loss_avg = [float(0.0) for i in range(params.get("num_levels"))]
        self.val_total = [float(0.0) for i in range(params.get("num_levels"))]

        self.t_loss_label = [('train loss ' + str(i)) for i in range(params.get("num_levels"))]
        self.v_loss_label = [('val loss ' + str(i)) for i in range(params.get("num_levels"))]
        self.t_acc_label = [('train acc ' + str(i)) for i in range(params.get("num_levels"))]
        self.v_acc_label = [('val acc ' + str(i)) for i in range(params.get("num_levels"))]


    def set_device(self, device):
        self.device = device

    def set_loaders(self, trainloader, valloader):
        self.trainloader = trainloader
        self.trainloader_iter  = iter(trainloader)

        self.valloader = valloader

    def next_batch(self):
        train_data = (self.trainloader_iter.next()).float()
        r = self.prep_batch(train_data)
        return r

    def prep_batch(self, train_data):
        try:
            if self.params.get('ndata') == "mnist":
                self.inputs, self.labels = (train_data[0].view(-1, 28 * 28)).to(self.device), train_data[1].to(
                    self.device)
            else:
                self.inputs, self.labels = train_data[0].to(self.device), train_data[1].to(self.device)
            return False
        except StopIteration:
            self.trainloader_iter = iter(self.trainloader)
            return True

    def do_stats(self, epoch, cycle_num):
        # reset stats before computation
        i = 0
        self.val_acc[i] = 0.0
        # self.val_outputs[i] = 0.0
        self.val_correct[i] = 0.0
        self.val_loss[i] = 0.0
        self.val_num[i] = 0
        self.val_loss_avg[i] = 0.0
        self.val_total[i] = 0.0

        # computing
        with torch.no_grad():
            for val_data in self.valloader:
                if self.params.get("ndata") == "mnist":
                    val_inputs, val_labels = (val_data[0].view(-1, 28 * 28)).to(self.device), val_data[1].to(self.device)
                else:
                    val_inputs, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)

                val_outputs = self.nets[i](val_inputs)
                self.val_loss[i] += self.criteria[i](val_outputs, val_labels)
                _, predicted = val_outputs.max(1)
                self.val_correct[i] += predicted.eq(val_labels).sum().item()
                self.val_total[i] += val_labels.size(0)
                self.val_num[i] += 1

            self.val_loss_avg[i] = self.val_loss[i].item() / self.val_num[i]
            self.val_acc[i] = self.val_correct[i] / self.val_total[i]
            self.train_acc[i] = self.train_correct[i] / self.train_total[i]

            wandb.log({'cycle': cycle_num, 'epoch': epoch, 'work': self.work, self.t_loss_label[i]: self.train_loss[i],
                       self.t_acc_label[i]: self.train_acc[i],
                       self.v_loss_label[i]: self.val_loss_avg[i], self.v_acc_label[i]: self.val_acc[i]})


    def train(self, epoch, params):
        self.trainloader_iter = iter(self.trainloader)
        # reset train stats
        self.train_total[0] = 0
        self.train_correct[0] = 0

        for train_data in self.trainloader_iter:
            if not self.prep_batch(train_data):
                self.labels_init = copy.deepcopy(self.labels)
                self.inputs_init = copy.deepcopy(self.inputs)

                if self.sgd_step():
                    self.do_stats(epoch=epoch, cycle_num=self.cycle_num)
                    return
                if self.cycle_num % self.sample_rate == 0:
                    self.do_stats(epoch=epoch,cycle_num=self.cycle_num)
                self.cycle_num += 1
            else:
                return

        for i, train_data in self.trainloader_iter:
            if not self.next_batch():
                self.sgd_step()
            else:
                return

    def sgd_step(self):
        self.optimizers[0].zero_grad()
        outputs = self.nets[0](self.inputs)
        self.train_loss[0] = self.criteria[0](outputs, self.labels)
        self.train_loss[0].backward()
        if not self.params.get('clip_max_norm') == 0:
            clip_grad_norm_(self.net.parameters(), self.params.get('clip_max_norm'), self.params.get('clip_norm_type'))
        self.optimizers[0].step()

        self.work += (self.params.get('num_layers')[0]) * (self.forward_cost + self.loss_cost + self.backprop_cost + self.step_cost)

        ## Loss/Accuracy Variables after one cycle
        with torch.no_grad():
            level = 0
            outputs = self.nets[0](self.inputs)
            loss = self.criteria[0](outputs, self.labels)
            _, self.train_predicted = outputs.max(1)
            self.train_total[0] += self.labels.size(0)
            self.train_correct[0] += self.train_predicted.eq(self.labels).sum().item()

        return False
