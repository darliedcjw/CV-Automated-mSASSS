import torch
from torch import nn

"""Fundaamental Blocks - Start"""
class Bottleneck(nn.Module):
    def __init__(self, in_filters, out_filters, expansion, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_filters * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out.clone() + residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, in_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_filters, in_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_filters)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out.clone() + x
        out = self.relu(out)

        return out


class ModuleBlock(nn.Module):
    def __init__(self, stage, out_branch, c):
        super().__init__()
        self.stage = stage
        self.out_branch = out_branch
        self.relu = nn.ReLU(inplace=True)
        
        self.branches = nn.ModuleList()
        for i in range(self.stage):
            in_filters = c*(2**i)
            branch = nn.Sequential(
                BasicBlock(in_filters=in_filters),
                BasicBlock(in_filters=in_filters),
                BasicBlock(in_filters=in_filters),
                BasicBlock(in_filters=in_filters)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for focus_branch in range(self.out_branch):
            self.fuse_layers.append(nn.ModuleList())
            
            for unfocus_branch in range(self.stage):
                if focus_branch == unfocus_branch:
                    # To Do Nothing
                    self.fuse_layers[-1].append(nn.Sequential())
                elif focus_branch < unfocus_branch:
                    # Upsampling
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c*(2**unfocus_branch), c*(2**focus_branch), kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(c*(2**focus_branch)),
                        nn.Upsample(scale_factor=(2**(unfocus_branch-focus_branch)), mode='nearest')
                    ))
                elif focus_branch > unfocus_branch:
                    # Downsampling
                    ops = []
                    for _ in range(focus_branch - unfocus_branch - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c*(2**unfocus_branch), c*(2**unfocus_branch), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c*(2**unfocus_branch)),
                            nn.ReLU(inplace=True)
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c*(2**unfocus_branch), c*(2**focus_branch), kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(c*(2**focus_branch))
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
            
    def forward(self, x):
        out = [branch(b) for branch, b in zip(self.branches, x)]
        out_fused = []
        for branch_ops in range(len(self.fuse_layers)):
            for branch in range(len(self.branches)):
                if branch == 0:
                    # Initialised the list
                    out_fused.append(self.fuse_layers[branch_ops][0](out[0]))
                else:
                    out_fused[branch_ops] = out_fused[branch_ops].clone() + self.fuse_layers[branch_ops][branch](out[branch])
        
        out = [self.relu(out) for out in out_fused]
        return out
"""Fundamental Blocks - End"""

"""HRnet - Start"""
class HRNet(nn.Module):
    def __init__(self, c=32, key=12):
        super().__init__()

        # Input (stem net)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 4, downsample=downsample),
            Bottleneck(256, 64, 4),
            Bottleneck(256, 64, 4),
            Bottleneck(256, 64, 4)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(nn.Sequential(  
                nn.Conv2d(256, c * (2 ** 1), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * (2 ** 1)),
                nn.ReLU(inplace=True)
            ))
        ])

 
        self.stage2 = nn.Sequential(
            ModuleBlock(stage=2, out_branch=2, c=c)
        )

        self.transition2 = nn.ModuleList([
            nn.Sequential(), 
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * (2 ** 2)),
                nn.ReLU(inplace=True)
            )) 
        ])

        self.stage3 = nn.Sequential(
            ModuleBlock(stage=3, out_branch=3, c=c),
            ModuleBlock(stage=3, out_branch=3, c=c),
            ModuleBlock(stage=3, out_branch=3, c=c),
            ModuleBlock(stage=3, out_branch=3, c=c)
        )

        self.transition3 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * (2 ** 3)),
                nn.ReLU(inplace=True)
            ))
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            ModuleBlock(stage=4, out_branch=4, c=c),
            ModuleBlock(stage=4, out_branch=4, c=c),
            ModuleBlock(stage=4, out_branch=1, c=c)
        )

        # Final layer (final_layer)
        self.final_layer = nn.Conv2d(c, key, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]

        x = self.stage2(x)

        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]

        x = self.stage3(x)
        
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]

        x = self.stage4(x)

        x = self.final_layer(x[0])

        return x



"""End"""

if __name__ == '__main__':
    x = torch.randn(1,3,256,192)
    # model = HRNet(32, 17, 0.1)
    model = HRNet(48, 17)
    model.load_state_dict(
        torch.load('./weights/pose_hrnet_w48_256x192.pth', map_location='cpu')
    )
    print('ok!!')

    out = model(x)
    print(out.shape)