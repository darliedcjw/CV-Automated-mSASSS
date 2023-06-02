import torch
import torch.nn as nn

torch.manual_seed(0)

class focal_loss(nn.Module):
    def __init__(self, 
        wf,
        fp):
        super().__init__()

        self.wf = wf
        self.fp = fp

    def forward(self, outputs, labels):
        outputs = nn.Softmax(dim=1)(outputs)
        loss = 0
        for index, output in enumerate(outputs): 
            class_idx = labels[index]
            p = output[class_idx]
            if class_idx == 0:
                loss += -(1 - self.wf)*(1 - p)**self.fp*torch.log(p)
            elif class_idx == 1:
                loss += -self.wf*(1 - p)**self.fp*torch.log(p)

        return loss / outputs.shape[0]
