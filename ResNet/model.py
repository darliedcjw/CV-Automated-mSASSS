import torch
import torch.nn as nn

torch.manual_seed(0)

resnet152_config = [
    (64, 7, 2), # Out, Kernel, Stride
    "MP",
    ["B", 3, False], # Residual Block, num_repeats, downsample
    ["B", 8, True], #8
    ["B", 36, True], #36
    ["B", 3, True], #3
    "AP", # Average Pooling
    "F", # Flattern
    ["L", 2048, 2], # FC, in, out
] 


# Basic CNN Block
class CNNBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        bn_act=True,
        **kwaargs):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwaargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.bn_act = bn_act

    def forward(self, x):
        if self.bn_act:
            x = self.conv(x)
            x = self.bn(x)
            x = self.leaky(x)
        else:
            x = self.conv(x)
        
        return x
            

# Residual Block (+)
class ResidualBlock(nn.Module):
    def __init__(self,
        channels,
        use_residual=True,
        num_repeats=1,
        downsample=False):
        super().__init__()
        
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        self.downsample = downsample

        self.layers = nn.ModuleList()

        if self.downsample:
            for num in range(num_repeats):
                self.layers += [nn.Sequential(
                        CNNBlock(channels*2 if num==0 else channels*4, channels, kernel_size=1, stride=2 if num==0 else 1),
                        CNNBlock(channels, channels, kernel_size=3, padding=1),
                        CNNBlock(channels, channels*4, kernel_size=1)
                )
            ]

        else:   
            for num in range(num_repeats):
                self.layers += [nn.Sequential(
                        CNNBlock(channels if num==0 else channels*4, channels, kernel_size=1),
                        CNNBlock(channels, channels, kernel_size=3, padding=1),
                        CNNBlock(channels, channels*4, kernel_size=1)
                    )
                ]
        
        
    def forward(self, x):
        for index, layer in enumerate(self.layers):
            if self.use_residual and index!=0:
                x = layer(x) + x
            else:
                x = layer(x)
        
        return x


class ResNet152(nn.Module):
    def __init__(self,
        in_channels=3,
        num_classes=20):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        

        self.sm = nn.Softmax(dim=1)


        self.layers = self._create_conv_layers()

        self.initialize_weights()


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        print("Loading ResNet152")
        for module in resnet152_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding = "same" if kernel_size == 3 else 3 if kernel_size == 7 else 0
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                if module[0] == "B":
                    num_repeats, downsample  = module[1:]
                    layers.append(
                        ResidualBlock(
                            in_channels,
                            num_repeats=num_repeats,
                            downsample=downsample
                        )
                    )
                    in_channels = in_channels*2

                elif module[0] == "L":
                    in_features, _ = module[1:]
                    layers.append(
                        nn.Linear(
                            in_features,
                            self.num_classes
                        )
                    )
            
            elif isinstance(module, str):
                if module == "MP":
                    layers.append(
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=2,
                            padding=1
                        )
                    )

                elif module == "AP":
                    layers.append(
                        nn.AdaptiveAvgPool2d(
                            output_size=(1,1)
                        )
                    )
                
                elif module == "F":
                    layers.append(
                        nn.Flatten()
                    )

        return layers


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.sm(x)         
        
        return x

    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    x = torch.randn(16,3,224,224)
    model = ResNet152(3, 2)
    out = model(x)
    a, b = torch.max(out, dim=1)
    
