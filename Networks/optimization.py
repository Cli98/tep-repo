from torch.optim import lr_scheduler, SGD


def setup_optimizer(para, opt):
    if opt.optimizer.lower() == "sgd":
        return SGD(para, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    raise NotImplementedError(
        'The optimizer you requested is not available, please contact developper'.format(opt.type.lower))


def setup_scheduler(optimizer, scheduler_type, gamma, delay_epochs, decay_iters):
    if scheduler_type == "lambda":
        # That's simply exp dacay
        # Sets the learning rate of each parameter group to the initial lr times a given function
        schedule_rule = lambda epoch: gamma ** ((epoch + 1) // delay_epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_rule)
        return scheduler
    if scheduler_type == "step":
        # gamma is not the same as the input
        # Assuming optimizer uses lr = 0.05 for all groups
        # lr = 0.05     if epoch < 30
        # lr = 0.005    if 30 <= epoch < 60
        # lr = 0.0005   if 60 <= epoch < 90
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_iters, gamma=0.1)
        return scheduler
    raise NotImplementedError(
        'The scheduler you requested is not available, please contact developper'.format(scheduler_type))


def require_grad(net,require_grad):
    # Nothing to do with optimization
    if not net:
        print("The network fails to initialize! Please double check!")
        return
    for para in net.parameters():
        para.requires_grad = require_grad

