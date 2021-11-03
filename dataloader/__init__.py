from dataloader.dataset import freespace
from dataloader.dataset_bdd import freespace_bdd
import torch

def CreateDataLoader(opt):
    dataset = freespace(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    return dataloader

def CreateDataLoader_bdd(opt):
    dataset = freespace_bdd(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    return dataloader
