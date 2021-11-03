import torch


def initialize_layer(layer, type = "normal", gain=0.02):
    classname = layer.__class__.__name__
    if hasattr(layer, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if type == 'normal':
            torch.nn.init.normal_(layer.weight.data, 0.0, gain)
        elif type == 'xavier':
            torch.nn.init.xavier_normal_(layer.weight.data, gain=gain)
        elif type == 'kaiming':
            torch.nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
        elif type == 'orthogonal':
            torch.nn.init.orthogonal_(layer.weight.data, gain=gain)
        elif type == 'pretrained':
            pass
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % type)
        if hasattr(layer, 'bias') and layer.bias is not None and type != 'pretrained':
            torch.nn.init.constant_(layer.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(layer.weight.data, 1.0, gain)
        torch.nn.init.constant_(layer.bias.data, 0.0)

def initialize_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # Update on 11/18/2020, initialization looks good.
    ret_net = None
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(int(gpu_ids[0]))
        ret_net = torch.nn.DataParallel(net, gpu_ids)
        net = ret_net.module

    pretrain_count, initial_count, total = 0, 0, 0
    for root_child in net.children():
        for children in root_child.children():
            total += 1
            if children in root_child.initialization_layer :
                initialize_layer(children, init_type, gain=init_gain)
                initial_count += 1
            else:
                initialize_layer(children, "pretrained", gain=init_gain)
                pretrain_count += 1
    print("Initialization complete for total of {} layers, where {} initialized with pretrain and {} initialized with "
          "indicated type of initialization".format(total, pretrain_count, initial_count))

    return ret_net if ret_net is not None else net