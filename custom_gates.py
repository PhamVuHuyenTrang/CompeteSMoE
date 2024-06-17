import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np
from fmoe.gates.base_gate import BaseGate

__all__ = [
    "CustomNaiveGate_Balance_SMoE",
    "CustomNaiveGate_Balance_XMoE",
    "CustomNaiveGate_Balance_StableMoE",
]


class CustomNaiveGate_Balance_SMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_blance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
class CustomNaiveGate_Balance_StableMoE(BaseGate):
    r"""
    Naive Gate StableMoE
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, d_model)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self._cosine(inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        eps1 = torch.randn(1) * 1e-4
        eps2 = torch.randn(1) * 1e-4
        mat1 = mat1 + eps1
        mat2 = mat2 + eps2
        mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
class MeanVarMLP(nn.Module):
    def __init__(self, d_model):
        super(MeanVarMLP, self).__init__()
        self.linear1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 4)
        self.register_parameter('weight1', self.linear1.weight)
        self.register_parameter('bias1', self.linear1.bias)
        self.register_parameter('weight2', self.linear2.weight)
        self.register_parameter('bias2', self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        mean_var = self.linear2(x)
        mean1, var1, mean2, var2 = torch.chunk(mean_var, 4, dim=-1)

        mean1 = torch.clamp(mean1, min=-0.05, max=0.05)
        mean2 = torch.clamp(mean2, min=-0.05, max=0.05) 
        var1 = torch.clamp(var1, min=0.01, max=0.1)  
        var2 = torch.clamp(var2, min=0.01, max=0.1)

        if torch.isnan(mean1).any() or torch.isnan(var1).any():
            print("NaN detected in MeanVarMLP forward pass (mean1 or var1)")
        if torch.isnan(mean2).any() or torch.isnan(var2).any():
            print("NaN detected in MeanVarMLP forward pass (mean2 or var2)")

        return mean1, mean2, var1, var2

class CustomNaiveGate_Balance_XMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        self.mean_var_mlp = MeanVarMLP(8)

        expert_embeddings = torch.empty(num_expert, 8)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False)

    def set_load_balance(self, gate, gate_top_k_idx):
        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)
    def forward(self, inp, return_all_scores=False):
        reduced_inp = self.inp_reduction(inp)
        gate = self._cosine(reduced_inp, self.expert_embeddings)
        #print("Gate values:", gate)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)
        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2):
        mean1, mean2, var1, var2 = self.mean_var_mlp(mat1)
        if torch.isnan(mean1).any() or torch.isnan(var1).any():
            raise ValueError("NaN found in mean1 or var1")
        if torch.isnan(mean2).any() or torch.isnan(var2).any():
            raise ValueError("NaN found in mean2 or var2")
    
        if torch.isinf(mean1).any() or torch.isinf(var1).any():
            raise ValueError("Infinity found in mean1 or var1")
        if torch.isinf(mean2).any() or torch.isinf(var2).any():
            raise ValueError("Infinity found in mean2 or var2")
        eps1 = self.reparameterize(mean1, torch.log(var1))
        eps1 = eps1.expand_as(mat1)
        eps2 = self.reparameterize(mean2, torch.log(var2))
        eps2 = torch.mean(eps2, dim = 0)
        eps2 = torch.ones_like(mat2) * eps2
        mat1 = self._normalize(mat1.float(), p=2.0, dim=1, eps=eps1)
        mat2 = self._normalize(mat2.float(), p=2.0, dim=1, eps=eps2)
        loss_vae_eps1 = -0.5 * torch.sum(1 + torch.log(var1) - mean1.pow(2) - torch.log(var1).exp())
        loss_vae_eps2 = -0.5 * torch.sum(1 + torch.log(var2) - mean2.pow(2) - torch.log(var2).exp())
        loss_vae = loss_vae_eps1 + loss_vae_eps2
        self.loss += loss_vae
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            scores[~ok] = scores[ok].min(dim = 0)
        return scores

    def _normalize(self, input, p: float = 2.0, dim: int = 1, eps: float = 1e-12):
        denom = input.norm(p, dim, keepdim=True).expand_as(input) + eps
        return input / denom

