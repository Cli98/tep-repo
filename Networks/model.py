import torch
import torch.nn as nn
from .backbone import backbone
from .header import header
from .initialization import initialize_net
from .optimization import setup_optimizer, require_grad, setup_scheduler
from parsers.parser import basic_parser, apply_parser, train_parser
import os


class Freespace_network(nn.Module):
    def __init__(self, num_layer, num_label):
        super(Freespace_network, self).__init__()
        self.backbone = backbone(num_layer)
        self.filters = self.backbone.filters
        self.header = header(self.filters, num_label)
        self.num_layer = num_layer
        self.num_label = num_label
        self.initialization_layers = self.header.initialization_layer.copy()

    def forward(self, feature_map):
        feature_map = self.backbone.backbone.conv1(feature_map)
        feature_map = self.backbone.backbone.bn1(feature_map)
        x1_0 = self.backbone.backbone.relu(feature_map)

        feature_map = self.backbone.backbone.maxpool(x1_0)
        x2_0 = self.backbone.backbone.layer1(feature_map)
        x3_0 = self.backbone.backbone.layer2(x2_0)
        x4_0 = self.backbone.backbone.layer3(x3_0)
        x5_0 = self.backbone.backbone.layer4(x4_0)

        # decoder
        x1_1 = self.header.conv1_1(torch.cat([x1_0, self.header.up2_0(x2_0)], dim=1))
        x2_1 = self.header.conv2_1(torch.cat([x2_0, self.header.up3_0(x3_0)], dim=1))
        x3_1 = self.header.conv3_1(torch.cat([x3_0, self.header.up4_0(x4_0)], dim=1))
        x4_1 = self.header.conv4_1(torch.cat([x4_0, self.header.up5_0(x5_0)], dim=1))

        x1_2 = self.header.conv1_2(torch.cat([x1_0, x1_1, self.header.up2_1(x2_1)], dim=1))
        x2_2 = self.header.conv2_2(torch.cat([x2_0, x2_1, self.header.up3_1(x3_1)], dim=1))
        x3_2 = self.header.conv3_2(torch.cat([x3_0, x3_1, self.header.up4_1(x4_1)], dim=1))

        x1_3 = self.header.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.header.up2_2(x2_2)], dim=1))
        x2_3 = self.header.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.header.up3_2(x3_2)], dim=1))

        x1_4 = self.header.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.header.up2_3(x2_3)], dim=1))
        out = self.header.final(x1_4)
        return out


class Freespace():
    def __init__(self, opt):
        self.net = Freespace_network(opt.num_layer, opt.num_labels)
        # After conclusion of call for initialization_net, the model will be transferred to GPU, if it is available.
        self.opt = opt
        if opt.gpu is not None:
            if isinstance(opt.gpu, int):
                opt.gpu = [opt.gpu]
        self.device = torch.device('cuda:{}'.format(opt.gpu[0])) if opt.gpu is not None else torch.device('cpu')
        self.gpu = opt.gpu
        print("current device is: {}".format(self.device))
        self.net = initialize_net(self.net, init_type='normal', init_gain=0.02, gpu_ids=opt.gpu)
        if opt.istrain:
            self.optimizer = setup_optimizer(self.net.parameters(), opt)
            require_grad(self.net, True)
            self.scheduler = setup_scheduler(self.optimizer, opt.scheduler_type, opt.lr_gamma,
                                              opt.lr_decay_epochs, opt.lr_decay_iters)

    def prepare_input(self, input):
        self.rgb_image = input['rgb_image'].to(self.device)
        self.label = input['label'].to(self.device)
        self.image_names = input['path']
        self.image_oriSize = input['oriSize']

    def forward(self):
        self.output = self.net(self.rgb_image)

    def get_loss(self):
        self.loss_segmentation = nn.CrossEntropyLoss()(self.output, self.label).to(self.device)

    def backward(self):
        self.loss_segmentation.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.get_loss()
        self.backward()
        self.optimizer.step()

    def get_current_losses(self):
        return self.loss_segmentation

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # save models to the disk
    def save_networks(self, epoch):
        save_filename = 'epoch_%s.pth' % (epoch)
        save_path = os.path.join(self.opt.checkpoint, save_filename)

        if isinstance(self.gpu, int):
            self.gpu = [self.gpu]
        if len(self.gpu) > 0 and torch.cuda.is_available():
            # torch.save(self.net.module.cpu().state_dict(), save_path)
            torch.save(self.net.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, epoch):
        load_filename = 'epoch_%s.pth' % (epoch)
        load_path = os.path.join(self.opt.checkpoint, load_filename)
        net = self.net
        # if isinstance(net, torch.nn.DataParallel):
        #     net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        for key in list(state_dict.keys()):
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)


if __name__ == "__main__":
    parser = train_parser()
    parser = apply_parser(parser)
    net = Freespace(parser)
