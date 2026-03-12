#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import math
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm_notebook
import json

import numpy as np

end_time = 6000
t_grid = np.linspace(0, end_time, 6001)

pars = dict(v=0.219, D=1/3, mu_a=0.01, beta=0.5493, c=1.0, ell=1000.0)
xc = np.array([10, 10, 16.0])  

def make_SD_pairs():
    xs = np.zeros((21,21,3))
    xd = np.zeros((21,21,3))
    for m in range(21):
        for n in range(21):
            xs[m,n] = [4+m, n, 0.0]
            xd[m,n] = [-4+m, n, 0.0]
    return xs, xd

xs_all, xd_all = make_SD_pairs()

Um_all = np.zeros((len(t_grid), 21, 21))

xs_all, xd_all = make_SD_pairs()
P = 21*21

xsP = xs_all.reshape(-1, 3)
xdP = xd_all.reshape(-1, 3)

v, D, mu_a, beta, c, ell =  (pars["v"], pars["D"], pars["mu_a"],
                                pars["beta"], pars["c"], pars["ell"])


def gl_nodes_weights_01(device, Q=120, dtype=torch.float32):
    u_nodes, u_weights = np.polynomial.legendre.leggauss(Q)  # (-1,1)
    u = 0.5 * (u_nodes + 1.0)   
    w = 0.5 * u_weights           
    u = torch.as_tensor(u, dtype=dtype, device=device)
    w = torch.as_tensor(w, dtype=dtype, device=device)
    return u, w  


def erfcx_torch(z):
    try:
        return torch.special.erfcx(z)
    except Exception:
        z = torch.clamp(z, min=1e-300)
        small = z < 25.0
        out = torch.empty_like(z)
        out[small] = torch.exp(z[small]**2) * torch.erfc(z[small])
        invz = 1.0 / z[~small]
        out[~small] = (invz / math.sqrt(math.pi)) * (1.0 - 0.5 * invz * invz)
        return out

def khat_batched_equiv(xc3_k, t_tq, v, D, beta, eps=1e-12):
    K = xc3_k.shape[0]
    out = torch.ones((K,)+t_tq.shape, dtype=t_tq.dtype, device=t_tq.device)

    mask = t_tq > 0
    if mask.any():
        tm = t_tq.clone()
        tm = torch.where(mask, tm, torch.ones_like(tm))  

        xc3 = xc3_k.view(K,1,1)
        tmv = tm.view(1, *tm.shape)

        denom = torch.sqrt(4.0 * v * D * tmv).clamp_min(eps)
        z = (xc3 + 2.0 * beta * v * D * tmv) / denom
        val = 1.0 - beta * torch.sqrt(math.pi * v * D * tmv) * erfcx_torch(z)

        out[:, mask] = val[:, mask]
    return out


def expo_terms(rcs2_kp, rdc2_kp, u, params, eps=1e-12):

    c = params.get("c", 1.0)
    D = params.get("D", 1/3)

    k, p = rcs2_kp.shape
    q = u.shape[0]
    rsum = (rcs2_kp + rdc2_kp).unsqueeze(-1)  

    expo_kpq = rsum.new_ones(k, p, q)  # [k,p,q]
    return expo_kpq


# In[6]:


def um_series_torch_chunked_equiv(
    t_grid, xs_all, xd_all, xc_batch,           
    c, v, D, mu_a, beta,
    quad_n=120,
    chunk_q=16, chunk_t=256, chunk_p=128, chunk_k=20,
    dtype=torch.float32, device="cuda",
    s_min=1e-18, eps=None,
):
    if eps is None:
        eps = torch.finfo(dtype).tiny

    t_full = torch.as_tensor(t_grid, dtype=dtype, device=device)  # [T]

    xs = torch.as_tensor(xs_all, dtype=dtype, device=device)
    xd = torch.as_tensor(xd_all, dtype=dtype, device=device)
    if xs.dim()==3: xs = xs.reshape(-1, xs.shape[-1])
    if xd.dim()==3: xd = xd.reshape(-1, xd.shape[-1])

    xc_batch = torch.as_tensor(xc_batch, dtype=dtype, device=device)    # [K,3]
    K, T, P = xc_batch.shape[0], t_full.shape[0], xs.shape[0]
    u, w = gl_nodes_weights_01(Q=quad_n, device=device, dtype=dtype)    # [Q],[Q]

    um = torch.zeros((K, T, P), dtype=dtype, device=device)

    pos = t_full > 0
    if not pos.any():
        return um
    idx_pos = torch.nonzero(pos, as_tuple=False).squeeze(1) 
    t_pos = t_full[pos]                                     

    pref_T = (c * torch.exp(-v * mu_a * t_pos)) / (16.0 * math.pi**3 * (D**2) * v)  

    for k0 in range(0, K, chunk_k):
        k1 = min(K, k0 + chunk_k)
        xcK = xc_batch[k0:k1]                   # [k,3]
        xc3_k = xcK[:, 2]                       # [k]

        rdx2 = torch.sum((xd[None,:,:] - xcK[:,None,:])**2, dim=2)
        rsx2 = torch.sum((xs[None,:,:] - xcK[:,None,:])**2, dim=2)

        for t0 in range(0, t_pos.numel(), chunk_t):
            t1 = min(t_pos.numel(), t0 + chunk_t)
            t = t_pos[t0:t1]                    # [t]

            for p0 in range(0, P, chunk_p):
                p1 = min(P, p0 + chunk_p)
                rdx2_kp = rdx2[:, p0:p1]        # [k,p]
                rsx2_kp = rsx2[:, p0:p1]        # [k,p]

                acc_k_t_p = torch.zeros((k1-k0, t.numel(), p1-p0), dtype=dtype, device=device)

                for q0 in range(0, u.numel(), chunk_q):
                    q1 = min(u.numel(), q0 + chunk_q)
                    uq = u[q0:q1]               # [q]
                    wq = w[q0:q1]               # [q]


                    s_TQ  = t[:,None] * uq[None,:]              # [t,q]
                    dt_TQ = (t[:,None] - s_TQ).clamp_min(eps)   # [t,q]


                    safe = (s_TQ > s_min) & (dt_TQ > s_min)     # [t,q]
                    factor_TQ = torch.zeros_like(s_TQ)
                    factor_TQ[safe] = torch.pow((dt_TQ[safe] * s_TQ[safe]), -1.5)

                    # denom
                    denom1_TQ = (4.0 * v * D * dt_TQ).clamp_min(eps)
                    denom2_TQ = (4.0 * v * D * s_TQ ).clamp_min(eps)

                    exp1_kptq = torch.exp(- rdx2_kp[:, :, None, None] / denom1_TQ[None, None, :, :])
                    exp2_kptq = torch.exp(- rsx2_kp[:, :, None, None] / denom2_TQ[None, None, :, :])

                    kh1_kTQ = khat_batched_equiv(xc3_k, dt_TQ, v, D, beta)
                    kh2_kTQ = khat_batched_equiv(xc3_k, s_TQ,  v, D, beta)

                    integrand_kptq = exp1_kptq * exp2_kptq \
                                     * kh1_kTQ[:, None, :, :] * kh2_kTQ[:, None, :, :] \
                                     * factor_TQ[None, None, :, :]

                    part_kpt = torch.einsum('kptq,q->kpt', integrand_kptq, wq)

                    acc_k_t_p.add_(part_kpt.transpose(1, 2))

                    del s_TQ, dt_TQ, safe, factor_TQ, denom1_TQ, denom2_TQ
                    del exp1_kptq, exp2_kptq, kh1_kTQ, kh2_kTQ, integrand_kptq, part_kpt


                acc_k_t_p.mul_(t.view(1,-1,1))                  # * t

                acc_k_t_p.mul_(pref_T[t0:t1].view(1,-1,1))      # * pref_T

                um[k0:k1, idx_pos[t0:t1], p0:p1] = acc_k_t_p
                del acc_k_t_p
    return um  # [K,T,P]


# In[7]:


def um_series_torch_chunked_equiv_multi(
    t_grid, xs_all, xd_all, xc_batch,            
    c, v, D, mu_a, beta,
    quad_n=120,
    chunk_q=16, chunk_t=256, chunk_p=128, chunk_k=20,
    dtype=torch.float32, device="cuda",
    s_min=1e-18, eps=None,
    w_multi=None,                          
):

    import math
    if eps is None:
        eps = torch.finfo(dtype).tiny

    t_full = torch.as_tensor(t_grid, dtype=dtype, device=device)     
    xs = torch.as_tensor(xs_all, dtype=dtype, device=device)
    xd = torch.as_tensor(xd_all, dtype=dtype, device=device)
    if xs.dim()==3: xs = xs.reshape(-1, xs.shape[-1])                
    if xd.dim()==3: xd = xd.reshape(-1, xd.shape[-1])                  # [P,3]

    xc_batch = torch.as_tensor(xc_batch, dtype=dtype, device=device)   # [K,M,3]
    K, M, T, P = xc_batch.shape[0], xc_batch.shape[1], t_full.shape[0], xs.shape[0]

    if w_multi is None:
        w = torch.ones((K, M), dtype=dtype, device=device)
    else:
        w = torch.as_tensor(w_multi, dtype=dtype, device=device)
        if w.ndim == 0:
            w = w.expand(K, M)
        elif w.ndim == 1:
            assert w.shape[0] == K, "w_multi 1D면 길이는 K와 같아야 합니다."
            w = w[:, None].expand(K, M)
        else:
            assert w.shape == (K, M), "w_multi는 (K,M)이어야 합니다."


    um = torch.zeros((K, T, P), dtype=dtype, device=device)


    pos = t_full > 0
    if not pos.any():
        return um
    idx_pos = torch.nonzero(pos, as_tuple=False).squeeze(1) 
    t_pos = t_full[pos]                                  
    Tpos = t_pos.numel()


    pref_T = (c * torch.exp(-v * mu_a * t_pos)) / (16.0 * math.pi**3 * (D**2) * v) 

    u, wq_full = gl_nodes_weights_01(Q=quad_n, device=device, dtype=dtype)  # [Q],[Q]

    integral_list = []

    for k0 in range(0, K, chunk_k):
        k1 = min(K, k0 + chunk_k)
        xcK = xc_batch[k0:k1]             # [k,M,3]
        wK  = w[k0:k1]                    # [k,M]
        k = k1 - k0


        for t0 in range(0, Tpos, chunk_t):
            t1 = min(Tpos, t0 + chunk_t)
            t = t_pos[t0:t1]              # [t]
            pref_t = pref_T[t0:t1]        # [t]


            for p0 in range(0, P, chunk_p):
                p1 = min(P, p0 + chunk_p)

                acc_k_t_p = torch.zeros((k, t.numel(), p1 - p0), dtype=dtype, device=device)

                for m in range(M):
                    xcKm = xcK[:, m, :]                  
                    z_km = xcKm[:, 2]                 

                    rdx2_kp = torch.sum((xd[None, p0:p1, :] - xcKm[:, None, :])**2, dim=2)
                    rsx2_kp = torch.sum((xs[None, p0:p1, :] - xcKm[:, None, :])**2, dim=2)

                    for q0 in range(0, u.numel(), chunk_q):
                        q1 = min(u.numel(), q0 + chunk_q)
                        uq = u[q0:q1]                   
                        wq = wq_full[q0:q1]           

                        s_TQ  = t[:, None] * uq[None, :]
                        dt_TQ = (t[:, None] - s_TQ).clamp_min(eps)

                        safe = (s_TQ > s_min) & (dt_TQ > s_min)
                        factor_TQ = torch.zeros_like(s_TQ)

                        denom1_TQ = (4.0 * v * D * dt_TQ).clamp_min(eps)
                        denom2_TQ = (4.0 * v * D * s_TQ ).clamp_min(eps)

                        exp1_kptq = torch.exp(- rdx2_kp[:, :, None, None] / denom1_TQ[None, None, :, :])
                        exp2_kptq = torch.exp(- rsx2_kp[:, :, None, None] / denom2_TQ[None, None, :, :])

                        kh1_kTQ = khat_batched_equiv(z_km, dt_TQ, v, D, beta)
                        kh2_kTQ = khat_batched_equiv(z_km, s_TQ,  v, D, beta)

                        integrand_kptq = exp1_kptq * exp2_kptq \
                                        * kh1_kTQ[:, None, :, :] * kh2_kTQ[:, None, :, :] \
                                        * factor_TQ[None, None, :, :]

                        part_kpt = torch.einsum('kptq,q->kpt', integrand_kptq, wq)

                        acc_k_t_p.add_( (wK[:, m].view(k,1,1)) * part_kpt.transpose(1, 2) )
                        integral_list.append(integrand_kptq)


                        del s_TQ, dt_TQ, safe, factor_TQ, denom1_TQ, denom2_TQ
                        del exp1_kptq, exp2_kptq, kh1_kTQ, kh2_kTQ, integrand_kptq, part_kpt

                acc_k_t_p.mul_(t.view(1, -1, 1))           
                acc_k_t_p.mul_(pref_t.view(1, -1, 1))     

                um[k0:k1, idx_pos[t0:t1], p0:p1] = acc_k_t_p
                del acc_k_t_p

    return um, integral_list  # [K,T,P]




def Um_from_um_trapexp_torch_batch(t_grid, um_KTP, ell, dtype, device):
    t = torch.as_tensor(t_grid, dtype=dtype, device=device)
    K, T, P = um_KTP.shape
    ell = torch.as_tensor(ell, dtype=dtype, device=device)

    dt = t[1:] - t[:-1]
    dt_full = torch.zeros_like(t)
    dt_full[1:] = dt

    Ti = t[:, None]
    Tj = t[None, :]

    Delta1 = torch.clamp(Ti - Tj, min=0.0)
    k1 = torch.exp(-Delta1 / ell) / ell

    Tj_prev = torch.roll(t, shifts=1)
    Tj_prev[0] = t[0]
    Delta0 = torch.clamp(Ti - Tj_prev[None, :], min=0.0)
    k0 = torch.exp(-Delta0 / ell) / ell
    k0[:, 0] = 0.0

    tril_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
    k1 = torch.where(tril_mask, k1, torch.zeros_like(k1))
    k0 = torch.where(tril_mask, k0, torch.zeros_like(k0))

    W = 0.5 * (k0 + k1) * dt_full[None, :]  # [T,T]
    Um = torch.einsum('tj,kjp->ktp', W, um_KTP)
    return Um



def erfcx_torch(z):
    try:
        return torch.special.erfcx(z)
    except Exception:

        z = torch.clamp(z, min=1e-300)
        small = z < 25.0
        out = torch.empty_like(z)

        out[small] = torch.exp(z[small]**2) * torch.erfc(z[small])

        invz = 1.0 / z[~small]
        out[~small] = (invz / math.sqrt(math.pi)) * (1.0 - 0.5 * invz * invz)
        return out

# -----------------------------
def khat_torch(xc3, tau_TQ, v, D, beta):

    tau = tau_TQ
    tiny = torch.finfo(tau.dtype).tiny


    tm = torch.clamp(tau, min=tiny)  # 0 방지

    denom = torch.sqrt(4.0 * v * D * tm)               
    z = (xc3 + 2.0 * beta * v * D * tm) / denom       


    val = 1.0 - beta * torch.sqrt(math.pi * v * D * tm) * erfcx_torch(z)

    out = torch.where(tau > 0, val, torch.zeros_like(val))
    return out




def um_series_torch_multi(t_grid, xs, xd, xc_list, c, v, D, mu_a, beta, DEVICE, DTYPE=torch.float32,
                          w_multi=None, quad_n=120):



    t_full = torch.as_tensor(t_grid, dtype=DTYPE, device=DEVICE)   # [T]
    xs = torch.as_tensor(xs, dtype=DTYPE, device=DEVICE)       # [P,3]
    xd = torch.as_tensor(xd, dtype=DTYPE, device=DEVICE)       # [P,3]
    T, P = t_full.shape[0], xs.shape[0]


    xc = torch.stack([x.to(DTYPE).to(DEVICE) for x in xc_list])    # [K,3]
    K = xc.shape[0]

    if w_multi is None:
        w_multi = torch.ones(K, dtype=DTYPE, device=DEVICE)
    else:
        w_multi = torch.as_tensor(w_multi, dtype=DTYPE, device=DEVICE)

    um_TP = torch.zeros((T, P), dtype=DTYPE, device=DEVICE)
    pos = t_full > 0
    if not pos.any():
        return um_TP

    t = t_full[pos]
    Tpos = t.shape[0]

    # ---- Gauss–Legendre ----
    u, gw = gl_nodes_weights_01(Q=quad_n, device=DEVICE, dtype=DTYPE)
    s = t[:, None] * u[None, :]
    t_minus_s = t[:, None] - s

    eps = torch.finfo(DTYPE).tiny
    s_min = 1e-18
    safe = (s > s_min) & (t_minus_s > s_min)
    factor_TQ = torch.zeros_like(s)
    factor_TQ[safe] = torch.pow((t_minus_s[safe] * s[safe]), -1.5)

    denom1_TQ = 4.0 * v * D * torch.clamp(t_minus_s, min=eps)
    denom2_TQ = 4.0 * v * D * torch.clamp(s,         min=eps)

    total_integral_PT = torch.zeros((P, Tpos), dtype=DTYPE, device=DEVICE)

    integral_dict = {k: [] for k in range(K)}
    pref_T = (c * torch.exp(-v * mu_a * t)) / (16.0 * math.pi**3 * (D**2) * v)
    for k in range(K):
        xc_k = xc[k]
        rdx = torch.linalg.norm(xd - xc_k, dim=1)
        rsx = torch.linalg.norm(xs - xc_k, dim=1)
        rdx2, rsx2 = rdx**2, rsx**2

        exp1_PTQ = torch.exp(-rdx2[:, None, None] / denom1_TQ[None, :, :])
        exp2_PTQ = torch.exp(-rsx2[:, None, None] / denom2_TQ[None, :, :])

        kh1_TQ = khat_torch(xc_k[2], t_minus_s, v, D, beta)
        kh2_TQ = khat_torch(xc_k[2], s, v, D, beta)

        integrand_PTQ = factor_TQ[None, :, :] * exp1_PTQ * exp2_PTQ * kh1_TQ[None, :, :] * kh2_TQ[None, :, :]
        integral_PT = torch.sum(integrand_PTQ * gw[None, None, :], dim=2) * t[None, :]
        coffi_integral = pref_T[:, None] * integral_PT.transpose(0, 1).contiguous()
        total_integral_PT += w_multi[k] * integral_PT 

        integral_dict[k].append(coffi_integral.cpu().detach())


    pref_T = (c * torch.exp(-v * mu_a * t)) / (16.0 * math.pi**3 * (D**2) * v)
    um_pos_TP = pref_T[:, None] * total_integral_PT.transpose(0, 1).contiguous()
    um_TP[pos, :] = um_pos_TP

    return um_TP, integral_dict




def Um_from_um_trapexp_torch(t_grid, um_TP, ell, dtype, DEVICE):

    t = torch.as_tensor(t_grid, dtype=dtype, device=DEVICE)  # [T]
    T = t.shape[0]
    P = um_TP.shape[1]
    ell = torch.as_tensor(ell, dtype=dtype, device=DEVICE)


    dt = t[1:] - t[:-1]                               # [T-1]
    dt_full = torch.zeros_like(t)
    dt_full[1:] = dt

    Ti = t[:, None]        # [T,1]
    Tj = t[None, :]        # [1,T

    Delta1 = torch.clamp(Ti - Tj, min=0.0)
    k1 = torch.exp(-Delta1 / ell) / ell               # [T,T]

    Tj_prev = torch.roll(t, shifts=1)                 # [T]
    Tj_prev[0] = t[0]                                 
    Delta0 = torch.clamp(Ti - Tj_prev[None, :], min=0.0)
    k0 = torch.exp(-Delta0 / ell) / ell               # [T,T]
    k0[:, 0] = 0.0                                  

    tril_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=DEVICE))
    k1 = torch.where(tril_mask, k1, torch.zeros_like(k1))
    k0 = torch.where(tril_mask, k0, torch.zeros_like(k0))

    W = 0.5 * (k0 + k1) * dt_full[None, :]            # [T,T]

    Um_TP = W @ um_TP
    return Um_TP


def add_noise_relative_rms(U, noise_level=0.05):
    rms = torch.sqrt(torch.mean(U**2))
    sigma = noise_level * rms
    noise = sigma * torch.randn_like(U)
    return U + noise