import wandb
from utils import *
import math

class MGOPT:
    def __init__(self, device, nets, optimizers, criteria, params):
        self.nets = nets
        self.toplevel = len(nets) - 1
        self.num_levels = len(nets)

        self.x1 = []
        for i, net in enumerate(self.nets):
            self.x1.append(copy.deepcopy(net.save_weights_to_tensor()))

        self.x2 = []
        for i, net in enumerate(self.nets):
            self.x2.append(copy.deepcopy(net.save_weights_to_tensor()))

        self.weights = []
        for i, net in enumerate(self.nets):
            self.weights.append(copy.deepcopy(net.save_weights_to_tensor()))

        # cannot use grads to initialize - not set yet
        self.prev_grads = copy.deepcopy(self.weights)

        self.criteria = criteria
        self.optimizers = optimizers
        self.device = device
        self.params = params

        self.forward_cost = 0.0
        self.loss_cost = 0.0
        self.step_cost = 0.0
        self.backprop_cost = 1.0

        self.transfer_cost = 0.01

        self.cycle_num = 0
        self.cycles_tot = 1
        self.cycles_corr = 0

        self.cycle_time = 0.0
        self.work = 0.0
        self.initialized = False
        self.trainloader = []
        self.valloader = []
        self.device = 'cpu'

        self.inputs_init = None
        self.labels_init = None

        self.nlevels = params.get("num_levels")

        self.sample_rate = int(params.get("sample_rate"))
        self.val_acc = [float(0.0) for i in range(params.get("num_levels"))]
        self.val_correct = [float(0.0) for i in range(params.get("num_levels"))]
        self.val_loss = [float(0.0) for i in range(params.get("num_levels"))]
        self.val_num = [0 for i in range(params.get("num_levels"))]
        self.val_loss_avg = [float(0.0) for i in range(params.get("num_levels"))]
        self.val_total = [float(0.0) for i in range(params.get("num_levels"))]

        self.train_acc = [float(0.0) for i in range(self.num_levels)]
        self.train_total = [int(0.0) for i in range(self.num_levels)]
        self.train_correct = [int(0.0) for i in range(self.num_levels)]
        self.train_loss = [float(0.0) for i in range(self.num_levels)]
        self.loss_after_e = [float(0.0) for i in range(self.num_levels)]
        self.loss_before_e = [float(0.0) for i in range(self.num_levels)]
        self.loss_diff_e = [float(0.0) for i in range(self.num_levels)]
        self.alpha = [float(0.0) for i in range(self.num_levels)]

        self.v = [float(0.0) for i in range(self.num_levels)]
        self.t_loss_label = [('train loss ' + str(i)) for i in range(params.get("num_levels"))]
        self.v_loss_label = [('val loss ' + str(i)) for i in range(params.get("num_levels"))]
        self.t_acc_label = [('train acc ' + str(i)) for i in range(params.get("num_levels"))]
        self.v_acc_label = [('val acc ' + str(i)) for i in range(params.get("num_levels"))]
        self.alpha_label = [('alpha L' + str(i)) for i in range(params.get("num_levels"))]

    def set_loaders(self, trainloader, valloader):
        self.trainloader = trainloader
        self.trainloader_iter  = iter(trainloader)
        self.valloader = valloader

    def set_device(self, device):
        self.device  = device

    def compute_x_v(self, x, v):
        r = 0
        with torch.no_grad():
            for i in range(len(v)):
                r += torch.sum(v[i] * x[i])
        return float(r)

    def compute_norm_v(self, v):
        n = 0.0
        for el in enumerate(v):
            n += float(torch.norm(el[1]))
        return n

    def linesearch(self, level, v):
        alpha, tau, c = self.params.get('ls_init_alpha'), 0.5, self.params.get('ls_c')

        grad_f_x = self.nets[level]._gather_flat_grad() # needs to be stored here because we abuse it all the time later
        self.optimizers[level].zero_grad()
        self.nets[level].prolong_step_to_gradient(self.x1[level - 1], self.nets[level - 1], self.params.get('p_weight_type'))
        p = self.nets[level]._gather_flat_grad() # we stored p in the gradient !

        m = torch.dot(grad_f_x,p)
        t = float(c * m)
        self.work += (self.params.get('num_layers')[level]) * (self.transfer_cost + self.step_cost + self.forward_cost + self.loss_cost)

        for i in range(10):
            self.optimizers[level].param_groups[0]['lr'] = alpha
            self.optimizers[level].step()
            with torch.no_grad():
                outputs = self.nets[level](self.inputs)
                loss = float(self.criteria[level](outputs, self.labels))
                if level == self.toplevel or not self.params.get("use_coupling"):
                    self.loss_after_e[level] = loss
                else:
                    y = self.nets[level].save_weights_to_tensor()
                    self.loss_after_e[level] = float(loss) - self.compute_x_v(y, v)

            if (self.loss_before_e[level] - self.loss_after_e[level]) >= alpha * t:
                if self.params.get('verbose'):
                    # print('ls on level %i ended at %i with alpha=%f' % (level, i, alpha))
                    print('ls - level %i, step %i, alpha=%f, loss diff= %f, t=%f, ' % (level, i, alpha, (self.loss_after_e[level]-self.loss_before_e[level]),t))
                return
            else:
                alpha = tau * alpha
                self.nets[level].copy_tensor_to_weights(self.x2[level])
                if self.params.get('verbose'):
                    print('ls on level %i continues at step %i with alpha=%f' % (level, i, alpha))

            if i == 9:
                print('no suitable step on level %i found, take step 0' % (level))
                self.optimizers[level].param_groups[0]['lr'] = 0.0
                self.nets[level].copy_tensor_to_weights(self.x2[level])
                with torch.no_grad():
                    outputs = self.nets[level](self.inputs_init)
                    if level == self.toplevel:
                        loss = float(self.criteria[level](outputs, self.labels_init))
                        self.loss_after_e[level] = loss
                    else:
                        loss = self.criteria[level](outputs, self.labels_init)
                        y = self.nets[level].save_weights_to_tensor()
                        self.loss_after_e[level] = float(loss) - self.compute_x_v(y, v)
                return

    def do_stats(self, epoch, cycle_num):
        # reset stats before computation
        for i in range(self.nlevels):
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

                for i in range(self.nlevels):
                    val_outputs = self.nets[i](val_inputs)
                    self.val_loss[i] += self.criteria[i](val_outputs, val_labels)
                    _, predicted = val_outputs.max(1)
                    self.val_correct[i] += predicted.eq(val_labels).sum().item()
                    self.val_total[i] += val_labels.size(0)
                    self.val_num[i] += 1

            for i in range(self.nlevels):
                self.val_loss_avg[i] = self.val_loss[i].item() / self.val_num[i]
                self.val_acc[i] = self.val_correct[i] / self.val_total[i]
                self.train_acc[i] = self.train_correct[i] / self.train_total[i]

        # logging
        for i in range(self.nlevels):
            wandb.log({'cycle': cycle_num, 'epoch': epoch, 'work': self.work, self.t_loss_label[i]: self.train_loss[i], self.t_acc_label[i]: self.train_acc[i],
                self.v_loss_label[i]: self.val_loss_avg[i], self.v_acc_label[i]: self.val_acc[i]}, step=cycle_num)
            if i > 0:
                wandb.log({'cycle': cycle_num, 'epoch': epoch, self.alpha_label[i]: self.alpha[i]}, step=cycle_num)

            if i == self.toplevel:
                if math.isnan(self.loss_diff_e[i]):
                    self.loss_diff_e[i] = 3
                wandb.log({'cycle': cycle_num, 'epoch': epoch, 'work': self.work, "train loss": self.train_loss[i], "train acc": self.train_acc[i], "val loss": self.val_loss_avg[i], "val acc": self.val_acc[i], "top level e": self.loss_diff_e[i]}, step=cycle_num)

    def next_batch(self):
        train_data = self.trainloader_iter.next()
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

    def train(self, epoch, params):
        self.trainloader_iter  = iter(self.trainloader)
        for i in range(self.num_levels):
            self.train_total[i] = 0
            self.train_correct[i] = 0

        for train_data in self.trainloader_iter:
            if not self.prep_batch(train_data):
                self.labels_init = copy.deepcopy(self.labels)
                self.inputs_init = copy.deepcopy(self.inputs)

                if self.mgopt_cycle(params, params.get("num_levels") - 1):
                    self.do_stats(epoch=epoch, cycle_num=self.cycle_num)
                    return

                if self.cycle_num % self.sample_rate == 0:
                    self.do_stats(epoch=epoch, cycle_num=self.cycle_num)

                print('cycle: ', self.cycle_num)
                self.cycle_num += 1
            else:
                return


    ### mg/opt main cycle
    def mgopt_cycle(self, params, level):

        if self.toplevel > level:
            self.nets[level + 1].restrict_weights(self.nets[level])  # => x_1
            self.work += (self.params.get('num_layers')[level]) * self.transfer_cost
            self.x1[level] = self.nets[level].save_weights_to_tensor()  # used for 'e' on lower levels

        for i in range(self.params.get("iters")[2*level]): # loop over number of iters at that level
            self.optimizers[level].zero_grad()
            outputs = self.nets[level](self.inputs)
            loss = self.criteria[level](outputs, self.labels)
            loss.backward()
            if i == 0:
                self.train_loss[level] = float(loss)

            if self.toplevel > level:
                if i == 0:
                    self.v[level] = self.nets[level + 1].form_v(self.nets[level])
                self.nets[level].subtract_v_from_grad_f_H(self.v[level])

            self.optimizers[level].step() # => x_2
            self.work += (self.params.get('num_layers')[level]) * (self.forward_cost + self.loss_cost + self.backprop_cost + self.step_cost)

        self.x2[level] = self.nets[level].save_weights_to_tensor()  # used for 'e' on lower levels

        # compute fresh gradient after step (needed for > 2 levels and further recursion)
        outputs = self.nets[level](self.inputs)
        loss = self.criteria[level](outputs, self.labels)
        self.work += self.params.get('num_layers')[level] * (self.forward_cost + self.loss_cost)

        if level > 0:
            self.optimizers[level].zero_grad()
            loss.backward()
            self.work += self.params.get('num_layers')[level] * (self.backprop_cost)

        if level < self.toplevel:
                self.loss_before_e[level] = float(loss) - self.compute_x_v(self.x2[level], self.v[level])
        else:
            self.loss_before_e[level] = float(loss)

        # MG recursion
        if level > 0:
            if self.mgopt_cycle(params, level - 1):
                return True
            # applying e not needed on level 0
            store_lr = self.optimizers[level].param_groups[0]['lr']
            if params.get('use_ls'):
                self.linesearch(level=level, v=self.v[level])
            else:
                self.optimizers[level].zero_grad()
                self.nets[level].prolong_step_to_gradient(self.x1[level - 1], self.nets[level - 1])
                self.optimizers[level].param_groups[0]['lr'] = 1.0
                self.optimizers[level].step()
                self.work += (self.params.get('num_layers')[level]) * (self.transfer_cost + self.step_cost + self.forward_cost + self.loss_cost)

                with torch.no_grad():
                    outputs = self.nets[level](self.inputs_init)
                    if level == self.toplevel:
                        loss = float(self.criteria[level](outputs, self.labels_init))
                        self.loss_after_e[level] = loss
                    else:
                        loss = self.criteria[level](outputs, self.labels_init)
                        y = self.nets[level].save_weights_to_tensor()
                        self.loss_after_e[level] = float(loss) - self.compute_x_v(y, self.v[level])
            self.alpha[level] = self.optimizers[level].param_groups[0]['lr']
            self.optimizers[level].param_groups[0]['lr'] = store_lr
        else:
            self.loss_after_e[0] = self.loss_before_e[0]

        with torch.no_grad():
            self.loss_diff_e[level] = self.loss_after_e[level] - self.loss_before_e[level]
            if params.get('use_ls'):
                if self.loss_diff_e[level] > 0:
                    for l in range(level+1):
                        # restore all the coarser grid updates ...
                        self.nets[l].copy_tensor_to_weights(self.x2[l])
                        print('reset coarse level update on level %i' % (l))
                        self.alpha[l] = 0.0
                else:
                    print('good coarse level update on level %i' % (level))

        # postsmoothing
        for i in range(params.get("iters")[2*level +1]):
            self.optimizers[level].zero_grad()
            outputs = self.nets[level](self.inputs)
            loss = self.criteria[level](outputs, self.labels)
            loss.backward()

            if self.toplevel > level:
                self.nets[level].subtract_v_from_grad_f_H(self.v[level])
            self.optimizers[level].step()
            self.work += (self.params.get('num_layers')[level])*(self.forward_cost + self.loss_cost + self.backprop_cost + self.step_cost)
            # stats
            if level == self.toplevel:
                self.cycles_tot += 1

        with torch.no_grad():
            # Loss/Accuracy per level
            outputs = self.nets[level](self.inputs)
            loss = self.criteria[level](outputs, self.labels)
            _, self.train_predicted = outputs.max(1)
            self.train_total[level] += self.labels.size(0)
            self.train_correct[level] += self.train_predicted.eq(self.labels).sum().item()
            self.train_loss[level] = float(loss)

        return False