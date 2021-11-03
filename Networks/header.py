import torch.nn as nn
from .customer_module import stack_conv_layer, upsample_layer


class header(nn.Module):
    def __init__(self, filters, num_label = 2):
        super(header, self).__init__()
        self.conv1_1 = stack_conv_layer([filters[0]*2, filters[0], filters[0]])
        self.conv2_1 = stack_conv_layer([filters[1]*2, filters[1], filters[1]])
        self.conv3_1 = stack_conv_layer([filters[2]*2, filters[2], filters[2]])
        self.conv4_1 = stack_conv_layer([filters[3]*2, filters[3], filters[3]])

        self.conv1_2 = stack_conv_layer([filters[0]*3, filters[0], filters[0]])
        self.conv2_2 = stack_conv_layer([filters[1]*3, filters[1], filters[1]])
        self.conv3_2 = stack_conv_layer([filters[2]*3, filters[2], filters[2]])

        self.conv1_3 = stack_conv_layer([filters[0]*4, filters[0], filters[0]])
        self.conv2_3 = stack_conv_layer([filters[1]*4, filters[1], filters[1]])

        self.conv1_4 = stack_conv_layer([filters[0]*5, filters[0], filters[0]])

        self.up2_0 = upsample_layer([filters[1], filters[0]])
        self.up2_1 = upsample_layer([filters[1], filters[0]])
        self.up2_2 = upsample_layer([filters[1], filters[0]])
        self.up2_3 = upsample_layer([filters[1], filters[0]])

        self.up3_0 = upsample_layer([filters[2], filters[1]])
        self.up3_1 = upsample_layer([filters[2], filters[1]])
        self.up3_2 = upsample_layer([filters[2], filters[1]])

        self.up4_0 = upsample_layer([filters[3], filters[2]])
        self.up4_1 = upsample_layer([filters[3], filters[2]])

        self.up5_0 = upsample_layer([filters[4], filters[3]])

        self.final = upsample_layer([filters[0], num_label])

        self.initialization_layer = [self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv1_2,
                                    self.conv2_2, self.conv3_2, self.conv1_3, self.conv2_3, self.conv1_4,
                                    self.up2_0, self.up2_1, self.up2_2, self.up2_3, self.up3_0, self.up3_1,
                                    self.up3_2, self.up4_0, self.up4_1, self.up5_0, self.final]


    def forward(self):
        pass

