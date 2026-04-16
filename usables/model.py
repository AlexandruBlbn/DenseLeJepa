import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


encoder = timm.create_model('swinv2_tiny_window8_256', pretrained=False, num_classes=0)
