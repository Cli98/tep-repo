from .model import Freespace


def create_model(parser):
    net = Freespace(parser)
    return net
