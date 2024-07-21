import torch
import torch.nn as nn
import torch.nn.functional as function

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_max_pool, dropout, is_transition):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        if is_max_pool:
            self.transition_layer = nn.MaxPool2d(2, 2)
        else:
            self.transition_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.is_transition = is_transition
        
    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x= self.dropout(x)
        
        if self.is_transition:
            x1 = self.transition_layer(x)
        else:
            x1 = x

        return x1, x

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_transpose_conv, dropout):
        super().__init__()
        if is_transpose_conv:
            self.transition_layer = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, 
                                                       stride=2, padding=1, output_padding=1)
        else:
            self.transition_layer = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                                  nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1),
                                                  nn.BatchNorm2d(in_channels//2),
                                                  nn.ReLU())
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
        
    def forward(self, x, skip_layer_input):
        x = self.transition_layer(x)
        x = torch.cat([skip_layer_input, x], dim=1)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.dropout(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_filters, is_max_pool=True, is_transpose_conv=True, n_classes = 3):
        super().__init__()
        self.cblock1 = ContractingBlock(3, n_filters, dropout=0, is_max_pool=is_max_pool, is_transition=True)
        self.cblock2 = ContractingBlock(n_filters, n_filters*2, dropout=0, is_max_pool=is_max_pool, is_transition=True)
        self.cblock3 = ContractingBlock(n_filters*2, n_filters*4, dropout=0, is_max_pool=is_max_pool, is_transition=True)
        self.cblock4 = ContractingBlock(n_filters*4, n_filters*8, dropout=0.3, is_max_pool=is_max_pool, is_transition=True)
        self.cblock5 = ContractingBlock(n_filters*8, n_filters*16, dropout=0.3, is_max_pool=is_max_pool, is_transition=False)

        self.ublock6 = ExpandingBlock(n_filters*16, n_filters*8, dropout=0, is_transpose_conv=is_transpose_conv)
        self.ublock7 = ExpandingBlock(n_filters*8, n_filters*4, dropout=0, is_transpose_conv=is_transpose_conv)
        self.ublock8 = ExpandingBlock(n_filters*4, n_filters*2, dropout=0, is_transpose_conv=is_transpose_conv)
        self.ublock9 = ExpandingBlock(n_filters*2, n_filters, dropout=0, is_transpose_conv=is_transpose_conv)

        self.conv10 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(n_filters, n_classes, kernel_size=1)
        
    def forward(self, x):
        x, c1 = self.cblock1(x)
        x, c2 = self.cblock2(x)
        x, c3 = self.cblock3(x)
        x, c4 = self.cblock4(x)
        x, _ = self.cblock5(x)
        x = self.ublock6(x, c4)
        x = self.ublock7(x, c3)
        x = self.ublock8(x, c2)
        x = self.ublock9(x, c1)

        x = self.conv10(x)
        x = self.conv11(x)

        return x
    
    
class MulticlassDiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self, num_classes, softmax_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim
    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = logits
        if self.softmax_dim is not None:
            probabilities = nn.Softmax(dim=self.softmax_dim)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        
        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()
        
        mod_a = intersection.sum()
        mod_b = targets.numel()
        
        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss