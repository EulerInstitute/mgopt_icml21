from ml_mnist_flat import *
from ml_mnist1d_flat import *


def process_config(config):
    optimizer_type = []
    learning_rate = []
    momentum = []
    weight_decay = []
    iters = []
    opt_setups = str(config.optimizers).split(';')
    num_levels = len(opt_setups)

    for item in enumerate(opt_setups):
        print(item)
        splitted_item = str(item[1]).split(' ')
        optimizer_type.append(str(splitted_item[0]))
        learning_rate.append(float(splitted_item[1]))
        momentum.append(float(splitted_item[2]))
        weight_decay.append(float(splitted_item[3])*2**((item[0]+1)-num_levels))

        iters.append(int(splitted_item[4]))
        iters.append(int(splitted_item[5]))
        print("Level Optimizer Type: %s  Learning Rate: %f Momentum: %f Weight Decay: %f pre-steps: %i post-steps: %i"
              % (optimizer_type[item[0]], learning_rate[item[0]], momentum[item[0]], weight_decay[item[0]], iters[2*item[0]], iters[2*item[0]+1]))

    resnet_blks_setup = []
    num_layers_fine =0
    for item in enumerate(str(config.resnet_blks).split(' ')):
        resnet_blks_setup.append(int(item[1]))
        num_layers_fine += int(item[1])

    num_layers = [int(num_layers_fine / (2 ** i)) for i in reversed(range(len(opt_setups)))]
    print("num_layers=", num_layers)

    scaling = [(1 / num_layers[i]) for i in range(len(opt_setups))]
    print("scaling=", scaling)

    params = dict(
        ndata=config.ndata,
        num_layers= num_layers,
        batch_size=config.batch_size,
        num_levels=len(opt_setups),
        use_ls=config.use_ls,
        num_epochs=config.num_epochs,
        training_type=config.training_type,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        iters=iters,
        debug_true=config.debug_true,
        device=config.device,
        nn_type=config.nn_type,
        sample_rate=config.sample_rate,
        target_acc=config.target_acc,
        resnet_blks_setup=resnet_blks_setup,
        scaling=scaling,
        verbose=config.verbose,
        ls_init_alpha=config.ls_init_alpha,
        ls_c = config.ls_c,
        r_scaling=config.r_scaling
    )
    return params

def initialize_nn(params, device):
    nets=[]
    if params.get("nn_type") == "mnist_flat":
        nets = [MLFlatNetMNIST(MNISTBlock,
                    num_layers=params.get('num_layers')[i],
                    scaling=params.get("scaling")[i],
                    width=10)
                for i in range(params.get("num_levels"))]
    elif params.get("nn_type") == "mnist1d_flat":
        nets = [MLFlatNetMNIST1d(MNIST1dBlock,
                     num_layers=params.get('num_layers')[i],
                     width=10,
                     scaling=params.get("scaling")[i])
                for i in range(params.get("num_levels"))]
    else:
        print("not implemented")
        exit(0)

    print("nets set up")
    for i in range(params.get("num_levels")):
        nets[i].to(device)
        print(nets[i])
    return nets

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def evalualate_testset(testloader, device, net, ndata='mnist'):
    with torch.no_grad():
        total = 0
        correct = 0
        for test_data in testloader:
            if ndata == 'mnist':
                test_images, test_labels = (test_data[0].view(-1, 28 * 28)).to(device), test_data[1].to(device)
            elif ndata =='cifar10':
                test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            outputs = net(test_images)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

    print('Test accuracy: %f' % (100 * correct / total))
