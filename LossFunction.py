# load general packages and functions
import torch


def graph_loss_calculator(output, target_output):
 
    LogSoftmax = torch.nn.LogSoftmax(dim=1)

    output = LogSoftmax(output)

    target_output = target_output/torch.sum(target_output, dim=1, keepdim=True)

    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    loss = criterion(target=target_output, input=output)

    return loss
