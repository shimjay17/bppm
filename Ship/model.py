import torch.nn as nn
# from dict import last_i 

# final_class= last_i + 1

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
    # This model replicates the bottleneck building block, used in resnet 50/101/152. 
    # basiclaly downscales  data using 1x1 convolution, then performs the main 3x3 convolution then upscales back up using 1x1 convolution.
    # This allows for more computational headroom
        super(block, self).__init__()
        self.expansion = 4
        # increase size 64>256>512 etc? need check
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # initializing, first convolution layer. 1x1 convolution, downscales the input data
        self.bn1 = nn.BatchNorm2d(out_channels)
        # batch normalization of the outhput of the first convolution. 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # the "main convolution". a 3x3 convolution is used.
        self.bn2 = nn.BatchNorm2d(out_channels)
        # batch normalization of output of the second convolution
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        #3rd convolution. upscales the data back up again from 64>256, hence self.expansion. 
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        #activation function
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        #sets x a sidentity
        
        x = self.conv1(x)
        # print('6 : ', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)


        # print('7 : ', x.shape)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        # print('8 : ', x.shape)
        x = self.bn3(x)
        # above performs necessary steps for bottleneck building block
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        # print('9 : ', x.shape)
        # add the initial value of x, saved as variable identity to current value of x.
        x = self.relu(x)
        # activation fuction
         
        return x

class ResNet(nn.Module):
    
    def __init__(self, block, layers, image_channels, final_class):
    # layers = how many layers in resnet (3, 4, 6,3 etc.)
    # image channels = number of channels in the input (eg. image has 3 channels, RGB). here, image channel is not specified so that any image of different channels can be used.
        super(ResNet, self).__init__()
        self.in_channels = 16
        # initial channel size is 16
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=7, stride=2, padding=1)
        # initial convolution. 
        self.bn1 = nn.BatchNorm2d(16)
        # normalize output
        self.relu = nn.ReLU()
        # activation function
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #pooling with a stride of 2

        self.dropout = nn.Dropout(0.5)
        
        # Resnet Layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=16, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=32, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=64, stride=2)
        self.layer4 = self._make_layer(block, layers[2], out_channels=128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # self.fc = nn.Linear(128*4, num_classes)

        self.fc = nn.Linear(128*4, 186)

        # self.fc2 = nn.Linear(num_classes, )
        
        
    def forward(self, x):
        # Here
        # breakpoint()
        # print('1: ', x.shape)
        x = self.conv1(x)
        # print('2 : ', x.shape)
        # Here
        x = self.bn1(x)
        # print('3 : ', x.shape)
        # Here
        x = self.relu(x)
        # print('4 : ', x.shape)
        # Here
        x = self.maxpool(x)
        # print('5 : ', x.shape)
        # Here
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Here
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)

        x = self.fc(x)
        
        return x
        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride !=1 or self.in_channels != out_channels*4:
        # if fuction necessary for when the channel size changes. if this occurs, the no of channels will need to be changes so that it can bee added back to x above
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*4))
            
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        # when the line above is executed, then the out_channels will be 256, as in class block, out_channels was multiplied by 4.
        self.in_channels = out_channels*4
        # then the in_channels need to be equal to the out_channels, hence *4. out_channel=64, need to multiply 64 by 4 to input 256 into in_channel.
        
        for i in range(num_residual_blocks -1):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 2)


        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x