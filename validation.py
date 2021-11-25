import os
from parsers.parser import validation_parser, summarize_parser, apply_parser
from dataloader import CreateDataLoader_bdd
from Networks import create_model
from utils.util import confusion_matrix, getScores, save_images
import torch
import numpy as np
import cv2
import torch.onnx

"""
python3 validation.py --dataroot ./datasets/Bdd --mode test --gpu 0 --image-type jpg 
--label-type png --epoch 2 --Width 640 --Height 384 --batch_size 1
"""
if __name__ == '__main__':
    valid_opt = apply_parser(validation_parser())
    valid_opt.isTrain = False
    summarize_parser(valid_opt)

    dataset = CreateDataLoader_bdd(valid_opt)
    model = create_model(valid_opt)
    model.load_networks(valid_opt.epoch)
    model.net.eval()

    test_loss_iter = []
    epoch_iter = 0
    conf_mat = np.zeros((valid_opt.num_labels, valid_opt.num_labels), dtype=np.float)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.prepare_input(data)
            if i==0:
                torch.onnx.export(model.net.module,  # model being run
                                  # [model.rgb_image, model.label, model.image_names, model.image_oriSize],
                                  model.rgb_image,
                                  # model input (or a tuple for multiple inputs)
                                  "freespace.onnx",
                                  # where to save the model (can be a file or file-like object)
                                  export_params=True,  # store the trained parameter weights inside the model file
                                  opset_version=11,  # the ONNX version to export the model to
                                  do_constant_folding=True,  # whether to execute constant folding for optimization
                                  input_names=['input'],  # the model's input names
                                  output_names=['output'],  # the model's output names
                                  dynamic_axes={'rgb_image': {0: 'batch_size', 1: "channel", 2: "height", 3: "width"},
                                                # 'label': {0: 'batch_size', 1: "height", 2: "width"},
                                                # 'path': {0: "file-path"},
                                                # 'oriSize': {0: "ori_width", 1: "ori_height"},
                                                'output': {0: 'batch_size'}}
                                )

            model.forward()
            model.get_loss()
            epoch_iter += valid_opt.batch_size
            gt = model.label.cpu().int().numpy()
            _, pred = torch.max(model.output.data.cpu(), 1)
            pred = pred.float().detach().int().numpy()

            # Resize images to the original size for evaluation
            oriSize = model.image_oriSize
            oriSize = int(oriSize[0]), int(oriSize[1])
            gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST),
                                  axis=0)
            conf_mat += confusion_matrix(gt, pred, valid_opt.num_labels)

            test_loss_iter.append(model.loss_segmentation)
            print('Epoch {0:}, iters: {1:}/{2:}, loss: {3:.3f} '.format(valid_opt.epoch,
                                                                        epoch_iter,
                                                                        len(dataset) * valid_opt.batch_size,
                                                                        test_loss_iter[-1]), end='\r')

        avg_test_loss = torch.mean(torch.stack(test_loss_iter))
        print('Epoch {0:} test loss: {1:.3f} '.format(valid_opt.epoch, avg_test_loss))
        globalacc, pre, recall, F_score, iou = getScores(conf_mat)
        print('Epoch {0:} glob acc : {1:.3f}, pre : {2:.3f}, recall : {3:.3f}, F_score : {4:.3f}, IoU : {5:.3f}'.format(
            valid_opt.epoch, globalacc, pre, recall, F_score, iou))
