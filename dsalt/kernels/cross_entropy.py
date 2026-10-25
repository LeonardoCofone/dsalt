# Adapted from Liger-Kernel (https://github.com/linkedin/Liger-Kernel).
#
# Copyright 2024 LinkedIn Corporation
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or
#    other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

import functools
import importlib
import operator

from typing import Callable

import torch
import triton
import triton.language as tl

from packaging.version import Version


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def is_npu_available() -> bool:
    try:
        from transformers.utils import is_torch_npu_available

        return is_torch_npu_available()
    except Exception:
        return False

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


def infer_device():
    if torch.cuda.is_available():
        return "cuda"
    elif is_npu_available():
        return "npu"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

def is_hip() -> bool:
    return torch.version.hip is not None


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps



def get_amp_custom_fwd_bwd() -> Callable:
    device = infer_device()
    if compare_version("torch", operator.ge, "2.4.0"):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    if hasattr(torch, "npu") and getattr(torch.npu, "amp", None) is not None:
        return torch.npu.amp.custom_fwd, torch.npu.amp.custom_bwd
    return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd


amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()


torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    token_accuracy_ptr,
    token_accuracy_stride,
    predicted_tokens_ptr,
    predicted_tokens_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    RETURN_TOKEN_ACCURACY: tl.constexpr,
    RETURN_PREDICTED_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
):

    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        if RETURN_TOKEN_ACCURACY:
            token_accuracy_ptr += program_id * token_accuracy_stride
            tl.store(token_accuracy_ptr, 0.0)
        if RETURN_PREDICTED_TOKENS:
            predicted_tokens_ptr += program_id * predicted_tokens_stride
            tl.store(predicted_tokens_ptr, -1)
        return

    loss_ptr += program_id * loss_stride
    if RETURN_Z_LOSS:
        z_loss_ptr += program_id * loss_stride
    if RETURN_TOKEN_ACCURACY:
        token_accuracy_ptr += program_id * token_accuracy_stride
    if RETURN_PREDICTED_TOKENS:
        predicted_tokens_ptr += program_id * predicted_tokens_stride

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    m = float("-inf")
    d = 0.0 
    argmax_idx = 0 
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32) 
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)

    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)

        if RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS:
            is_max_mask = X_block == block_max
            masked_offsets = tl.where(is_max_mask, X_offsets, n_cols)
            current_block_argmax_idx = tl.min(masked_offsets)

            is_new_max = block_max > m
            argmax_idx = tl.where(is_new_max, current_block_argmax_idx, argmax_idx)

        if label_smoothing > 0:
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block * weight_block, 0.0))
            else:
                scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
            ).cast(tl.float32)
            if HAS_SOFTCAPPING:
                intermediate = tanh(X_block / softcap)
                X_block = softcap * intermediate

            if not HAS_WEIGHT:
                X_block = tl.exp(X_block - m) / d
                X_block += 2 * lse_square_scale * lse * X_block
                X_block += -eps
                X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
                if reduction == "mean":
                    X_block = X_block / n_non_ignore
            else:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                softmax_X = tl.exp(X_block - m) / d
                dloss_ori = (1 - label_smoothing) * softmax_X
                dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
                dloss_ori = dloss_ori * weight_y
                dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
                dz_loss = 2 * lse_square_scale * lse * softmax_X
                if reduction == "mean":
                    dloss_ori = dloss_ori / sum_non_ignore_weight
                    dloss_smooth = dloss_smooth / sum_non_ignore_weight
                    dz_loss = dz_loss / n_non_ignore
                X_block = dloss_ori + dloss_smooth + dz_loss

            if HAS_SOFTCAPPING:
                X_block = X_block * (1 - intermediate * intermediate)

            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    tl.debug_barrier()

    loss = lse - ori_X_y
    if HAS_WEIGHT:
        loss = weight_y * loss

    if label_smoothing > 0:
        if HAS_WEIGHT:
            smooth_loss = scaled_x_sum + eps * lse * weight_sum
        else:
            smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss

    z_loss = lse_square_scale * lse * lse
    if reduction == "mean":
        if HAS_WEIGHT:
            loss = loss / sum_non_ignore_weight
        else:
            loss = loss / n_non_ignore
        z_loss = z_loss / n_non_ignore
    loss += z_loss

    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS:
        tl.store(z_loss_ptr, z_loss)
    if RETURN_TOKEN_ACCURACY:
        is_correct = 1.0 if argmax_idx == y else 0.0
        tl.store(token_accuracy_ptr, is_correct)
    if RETURN_PREDICTED_TOKENS:
        tl.store(predicted_tokens_ptr, argmax_idx)

if infer_device() == "xpu":
    MAX_FUSED_SIZE = 4096
elif infer_device() == "npu":
    MAX_FUSED_SIZE = 2048
else:
    MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    
    program_id = tl.program_id(0).to(tl.int64)

    X_ptr += program_id * X_stride

    grad_output = tl.load(grad_output_ptr)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


def get_npu_core_count(default: int = 20) -> int:
    try:
        utils = triton.runtime.driver.active.utils
        props = utils.get_device_properties(0)
        return int(props.get("num_vectorcore", default))
    except Exception:
        return default


def set_large_grf_mode(kernel_args: dict):
    if compare_version("pytorch-triton-xpu", operator.ge, "3.6.0") or compare_version("triton", operator.ge, "3.6.0"):
        kernel_args["grf_mode"] = "256"
    else:
        kernel_args["grf_mode"] = "large"

MAX_FUSED_SIZE = 2048 if infer_device() == "npu" else 65536 // 2


def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
    accum_dtype=None,
    use_token_scaling=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"
    )
    device = _input.device

    input_requires_grad = _input.requires_grad

    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H) 
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size) 

    grad_input = torch.zeros_like(_input, device=device)

    if input_requires_grad:
        if accum_dtype is None:
            grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
            grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
        else:
            grad_weight = torch.zeros_like(weight, dtype=accum_dtype, device=device) if weight.requires_grad else None
            grad_bias = torch.zeros_like(bias, dtype=accum_dtype, device=device) if bias is not None else None
    else:
        grad_weight = None
        grad_bias = None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    token_accuracy_1d = torch.zeros(BT, dtype=torch.float32, device=device) if return_token_accuracy else None
    predicted_tokens_1d = torch.full((BT,), -1, dtype=torch.int64, device=device) if return_predicted_tokens else None

    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]

        logits_chunk = _input_chunk @ weight.t()  
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]

        n_rows = logits_chunk.shape[0]

        if use_token_scaling:
            logits_for_softmax = logits_chunk.detach().clone()
            if softcap is not None:
                logits_for_softmax = softcap * torch.tanh(logits_for_softmax / softcap)

            probs = torch.softmax(logits_for_softmax, dim=-1)

            valid_target_mask = target_chunk != ignore_index
            valid_targets = target_chunk[valid_target_mask]

            if len(valid_targets) > 0:
                valid_probs = probs[valid_target_mask]
                pred_probs_valid = torch.gather(valid_probs, -1, valid_targets.unsqueeze(-1)).squeeze(-1)

                pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype, device=probs.device)
                pred_probs[valid_target_mask] = pred_probs_valid
            else:
                pred_probs = torch.zeros_like(target_chunk, dtype=probs.dtype, device=probs.device)

            scaling_factors = pred_probs.detach()  

        loss_1d_slice = loss_1d[start_idx:end_idx] 
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None
        token_accuracy_1d_slice = token_accuracy_1d[start_idx:end_idx] if return_token_accuracy else None
        predicted_tokens_1d_slice = predicted_tokens_1d[start_idx:end_idx] if return_predicted_tokens else None

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1), 
            weight_ptr=ce_weight,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1), 
            token_accuracy_ptr=token_accuracy_1d_slice,
            token_accuracy_stride=token_accuracy_1d_slice.stride(-1)
            if return_token_accuracy
            else 0, 
            predicted_tokens_ptr=predicted_tokens_1d_slice,
            predicted_tokens_stride=predicted_tokens_1d_slice.stride(-1)
            if return_predicted_tokens
            else 0,
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
            weight_sum=ce_weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            RETURN_TOKEN_ACCURACY=return_token_accuracy,
            RETURN_PREDICTED_TOKENS=return_predicted_tokens,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            HAS_GRADIENTS=input_requires_grad,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        if use_token_scaling:
            loss_1d_slice = loss_1d_slice * scaling_factors
            if return_z_loss:
                z_loss_1d_slice = z_loss_1d_slice * scaling_factors

        loss_1d[start_idx:end_idx] = loss_1d_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice
        if return_token_accuracy:
            token_accuracy_1d[start_idx:end_idx] = token_accuracy_1d_slice
        if return_predicted_tokens:
            predicted_tokens_1d[start_idx:end_idx] = predicted_tokens_1d_slice
        grad_logits_chunk = logits_chunk

        if use_token_scaling:
            scaling_factors_expanded = scaling_factors.unsqueeze(-1) 
            grad_logits_chunk = grad_logits_chunk * scaling_factors_expanded

        if input_requires_grad:
            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None and input_requires_grad:
            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()

        if bias is not None and input_requires_grad:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        token_accuracy = torch.sum(token_accuracy_1d) / total_n_non_ignore if return_token_accuracy else None

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
        accum_dtype=None,
        use_token_scaling: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):

        loss, z_loss, token_accuracy, predicted_tokens, grad_input, grad_weight, grad_bias = (
            fused_linear_cross_entropy_forward(
                _input=_input,
                weight=weight,
                target=target,
                bias=bias,
                ce_weight=ce_weight,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                softcap=softcap,
                return_z_loss=return_z_loss,
                accum_dtype=accum_dtype,
                use_token_scaling=use_token_scaling,
                return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
            )
        )
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens
        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        if ctx.return_z_loss:
            del grad_output2
        if ctx.return_token_accuracy:
            del grad_output3
        if ctx.return_predicted_tokens:
            del grad_output4
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  
            None, 
            None,  
        )