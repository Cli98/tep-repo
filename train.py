import numpy as np
import random, torch, cv2, time
from dataloader import CreateDataLoader_bdd
from Networks import create_model
from utils.util import confusion_matrix, getScores, print_current_losses
from parsers.parser import summarize_parser, apply_parser, validation_parser, train_parser
from tensorboardX import SummaryWriter


"""
python3 train.py --dataroot ./datasets/Bdd/ --Width 640 --Height 384 --mode train --batch_size 20 --gpu 0,1,2,3
"""
if __name__ == "__main__":
    parser = train_parser()
    train_opt = apply_parser(parser)
    summarize_parser(train_opt)
    # For training mode, enforce "istrain" option as True
    train_opt.istrain = True

    train_dataset = CreateDataLoader_bdd(train_opt)
    train_dataset_size = len(train_dataset)
    print("Total number of training images = {}".format(train_dataset_size))

    valid_opt = validation_parser()
    valid_opt = apply_parser(valid_opt)
    valid_opt.istrain = False
    valid_opt.mode = "val"
    valid_opt.batch_size = 1

    valid_dataset = CreateDataLoader_bdd(valid_opt)
    valid_dataset_size = len(valid_dataset)
    print("Total number of validation images = {}".format(valid_dataset_size))
    writer = SummaryWriter()
    # Set seed to make sure that we can reproduce result
    if train_opt.seed:
        np.random.seed(train_opt.seed)
        random.seed(train_opt.seed)
        torch.manual_seed(train_opt.seed)
        torch.cuda.manual_seed(train_opt.seed)

    model = create_model(train_opt)
    if train_opt.continue_train:
        model.load_networks(train_opt.epoch)

    total_steps = 0
    tfcount = 0
    F_score_max = 0

    for epoch in range(train_opt.epoch, train_opt.max_epoch + 1):
        ### Training on the training set ###
        model.net.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        train_loss_iter = []
        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()
            if total_steps % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            model.prepare_input(data)
            model.optimize_parameters()

            if total_steps % train_opt.print_freq == 0:
                tfcount = tfcount + 1
                losses = model.get_current_losses()
                losses_val = losses.float().detach().cpu().numpy()
                train_loss_iter.append(losses_val)
                t = (time.time() - iter_start_time) / train_opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t, t_data)
                writer.add_scalar('train/whole_loss', losses_val, tfcount)
                # There are several whole_loss values shown in tensorboard in one epoch,
                # to help better see the optimization phase
                if total_steps % 2000 == 0:
                    # Enforce a checkpoint to avoid interruption.
                    model.save_networks('enforce_epoch_{}'.format(epoch_iter))

            iter_data_time = time.time()
        model.save_networks('{}'.format(epoch))

        print(
            'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.max_epoch, time.time() - epoch_start_time))
        model.update_learning_rate()

        mean_loss = np.mean(train_loss_iter)
        writer.add_scalar('train/mean_loss', mean_loss, epoch)

        ### Evaluation on the validation set ###
        model.net.eval()
        valid_loss_iter = []
        epoch_iter = 0
        conf_mat = np.zeros((valid_opt.num_labels, valid_opt.num_labels), dtype=np.float)
        with torch.no_grad():
            for i, data in enumerate(valid_dataset):
                model.prepare_input(data)
                model.forward()
                model.get_loss()
                epoch_iter += valid_opt.batch_size
                gt = model.label.cpu().int().numpy()
                _, pred = torch.max(model.output.data.cpu(), 1)
                pred = pred.float().detach().int().numpy()

                # Resize images to the original size for evaluation
                oriSize = model.image_oriSize
                oriSize = int(oriSize[0]), int(oriSize[1])
                gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST),
                                    axis=0)
                pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST),
                                      axis=0)

                conf_mat += confusion_matrix(gt, pred, valid_opt.num_labels)
                losses = model.get_current_losses()
                valid_loss_iter.append(losses)
                print('valid epoch {0:}, iters: {1:}/{2:} '.format(epoch, epoch_iter,
                                                                   len(valid_dataset) * valid_opt.batch_size), end='\r')

        avg_valid_loss = torch.mean(torch.stack(valid_loss_iter))
        globalacc, pre, recall, F_score, iou = getScores(conf_mat)
        print("Check validation loss, f-score, iou {}, {}, {}".format(avg_valid_loss, F_score, iou))
        writer.add_scalar('valid/loss', avg_valid_loss, epoch)
        writer.add_scalar('valid/global_acc', globalacc, epoch)
        writer.add_scalar('valid/pre', pre, epoch)
        writer.add_scalar('valid/recall', recall, epoch)
        writer.add_scalar('valid/F_score', F_score, epoch)
        writer.add_scalar('valid/iou', iou, epoch)
        # Save the best model according to the F-score, and record corresponding epoch number in tensorboard
        if F_score > F_score_max:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('best')
            F_score_max = F_score
            writer.add_text('best model', str(epoch))
