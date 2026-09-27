import torch

def select_targets(output: torch.Tensor, target_ind):
    if target_ind is None:
        return output
    # single integer target
    #TODO
    return output

def compute_grad(input, output, target):
        
        # if target is not None:
        #     output = select_targets(output.tensor,target)

        #     gradients = torch.autograd.grad(output, input.tensor,    grad_outputs=torch.ones_like(output))[0]
        # else:
        gradients = torch.autograd.grad(output.tensor, input.tensor,    grad_outputs=torch.ones_like(output.tensor))[0]

        return gradients
        

        
