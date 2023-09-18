import torch
import numpy as np
from scipy.ndimage import zoom
from IPython.core.debugger import set_trace
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


class MaxpoolFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding=1, dilation=1):
        _rank = input.dim()
        if _rank == 3:
            # 1d
            pool_func = F.max_pool1d_with_indices
        elif _rank == 4:
            # 2d
            pool_func = F.max_pool2d_with_indices
        elif _rank == 5:
            # 3d
            pool_func = F.max_pool3d_with_indices
        else:
            raise ValueError("Invalid input")

        output, indices = pool_func(
            input, kernel_size, stride, padding, dilation,
            return_indices=True)
        print("Input[{}]: ".format(input.shape), input)
        print("Out[{}]: ".format(output.shape), output)
        print("Max indices: ", indices)
        ctx._rank = _rank
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.expected_size = input.size()
        ctx.maximum_indices = indices
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _rank = ctx._rank
        if _rank == 3:
            # 1d
            unpool_func = F.max_unpool1d
        elif _rank == 4:
            # 2d
            unpool_func = F.max_unpool2d
        elif _rank == 5:
            # 3d
            unpool_func = F.max_unpool3d
        else:
            raise ValueError("Invalid input")

        grad_output_unpool = unpool_func(
            grad_output, ctx.maximum_indices, ctx.kernel_size, ctx.stride, ctx.padding,
            output_size=ctx.expected_size)
        print("Grad_output[{}]: ".format(grad_output.shape), grad_output)
        print("Grad_output_unpool[{}]: ".format(
            grad_output_unpool.shape), grad_output_unpool)
        return grad_output_unpool, None, None, None, None


class CustomMaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        return MaxpoolFunction.apply(x, self.kernel_size, self.stride, self.padding, self.dilation)


def generate_grad_cam(m_dict, c_index, activation_layer):
    y_c = m_dict['fc'][0, c_index]
    A_k = m_dict[activation_layer]

    grad_val = torch.autograd.grad(y_c, A_k, retain_graph=True)[0].data

    # move CUDA tensor to CPU
    A_k = A_k.cpu()
    grad_val = grad_val.cpu()

    # remove batch dim
    conv_out = A_k.data[0].numpy()
    grad_val = grad_val[0].numpy()

    weights = np.mean(grad_val, axis=(1, 2, 3))

    grad_cam = np.zeros(dtype=np.float32, shape=conv_out.shape[1:])
    for k, w in enumerate(weights):
        grad_cam += w * conv_out[k]

    # upsample grad_cam
    temporal_ratio = 178 / grad_cam.shape[0]
    spatial_ratio = 224 / grad_cam.shape[1]

    grad_cam = zoom(grad_cam, (temporal_ratio, spatial_ratio, spatial_ratio))

    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = grad_cam / (grad_cam.max((1, 2))+1e-6)[:, None, None]

    return grad_cam


class LR_Wramer():
    def __init__(self, optimizer, scheduler=None, until=2000):
        super().__init__()
        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.until = until
        self.global_step = 0

        self.scheduler = scheduler

    def step(self, epoch=None):
        # warm up lr
        if self.global_step < self.until:
            lr_scale = min(1.0, float(
                self.global_step + 1) / self.until)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_scale * self.base_lr
        else:
            if self.scheduler is not None:
                self.scheduler.step()

        self.global_step += 1
