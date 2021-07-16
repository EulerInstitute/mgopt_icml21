import time
from torch import optim
from train_mgopt import *
from data_utils import *
import wandb

print("run.py ....")

hyperparameter_defaults = dict(
    batch_size=1000,
    optimizers='sgd 0.1 0.0 0.0 1 0;sgd 0.1 0.0 0.0 1 1;sgd 0.1 0.0 0.0 1 1;sgd 0.1 0.0 0.0 1 0', # optimizer, lr, momentum, weightdecay, pre-, post-
    # optimizers='sgd 0.1 0.0 0.0 1 0;sgd 0.1 0.0 0.0 1 0', # optimizer, lr, momentum, weightdecay, pre-, post-
    # optimizers='sgd 0.01 0.0 0.0 1 0',
    num_epochs=3,
    random_seed=2536481,
    training_type='mgopt',
    ndata="mnist1d",
    # ndata="mnist",
    # debug_true=False,
    debug_true=True,
    use_batchnorm=True,
    # use_coupling=False,
    use_ls=False,
    # use_ls=True,
    # use_batchswitch=True,
    use_kgkwiring=True,
    # device='cuda:0',
    device='cpu',
    description='__',
    nn_type='mnist1d_flat',
    # nn_type='mnist_flat',
    resnet_blks='256',
    ls_init_alpha=1.0,
    ls_c=0.5,
    sample_rate=1,
    target_acc=0.99,
    r_scaling=0.66,
    verbose=True,
    use_wandb=True,
)

milestones = [50, 150]
wandb.init(config=hyperparameter_defaults)
config = wandb.config
params = process_config(config)
toplevel = params.get("num_levels") - 1
nlevels = params.get("num_levels")

torch.manual_seed(config.random_seed)
print('Initial seed:', config.random_seed)

wandb.run.name = config.description \
                 + "_" + str(config.training_type) \
                 + "_" + str(config.nn_type) \
                 + "_" + str(config.batch_size) \
                 + "_l" + str(params.get('num_layers')[toplevel]) \
                 + "_L" + str(params.get('num_levels')) \
                 + "_ls" + str(config.use_ls) \
                 + "_" + str(config.ls_init_alpha) \
                 + "_" + str(config.ls_c) \
                 + "_lr" + str(params.get('learning_rate')[toplevel]) \
                 + "_m" + str(params.get('momentum')[toplevel]) \
                 + "_wd" + str(params.get("weight_decay")[toplevel]) \
                 + "_rs" + str(config.random_seed)

wandb.run.save()

# GPU access, but mnist_flat doesnt need that
if params.get("nn_type") == "mnist_flat":
    device = 'cpu'
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # printing gpu/cpu

trainloader, valloader, testloader, classes = get_data(config.ndata, config.batch_size, params.get("debug_true"))
nets = initialize_nn(params, device)
criteria = [nn.CrossEntropyLoss() for i in range(params.get('num_levels'))]

optimizers = []
opt_types = params.get('optimizer_type')
for i in range(len(opt_types)):
    if opt_types[i] == 'sgd':
        optimizers.append(optim.SGD(nets[i].parameters(),
                        lr=float(params.get('learning_rate')[i]),
                        momentum=float(params.get('momentum')[i]),
                        weight_decay=float(params.get("weight_decay")[i])))
    elif opt_types[i] == 'adam':
        optimizers.append(optim.Adam(nets[i].parameters(),
                        lr=float(params.get('learning_rate')[i]),
                        weight_decay=float(params.get("weight_decay")[i])))


schedulers = []
if params.get('training_type') == 'mgopt':
    training_type = MGOPT(device, nets, optimizers, criteria, params)
    for i in range(params.get('num_levels')):
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
            optimizers[i],
            milestones=milestones,
            gamma=0.1
        ))
else:
    print("No training type defined.")
    exit(0)

training_type.set_loaders(trainloader, valloader)
training_type.set_device(device)

batch_num = 0
start_t = time.time()  # for recording training time
for epoch in range(params.get('num_epochs')):  # loop over the dataset multiple times
    training_type.train(epoch=epoch, params=params)
    for i in range(nlevels):
        schedulers[i].step()
        print('epoch= %i, level %i, lr=%f' % (epoch, i, optimizers[i].param_groups[0]['lr']))

    print('epoch %d :  val loss: %.3f val acc: %.3f' % (epoch, training_type.val_loss_avg[toplevel], training_type.val_acc[toplevel]))

print("Training time: ", time.time() - start_t)