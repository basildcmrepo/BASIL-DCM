#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import numpy as np
import torch
from typing import Optional


def spm_csd_analytic_torch(
    A: torch.Tensor,         # (B, n, n)
    a: torch.Tensor,         # (B, 2)
    b: torch.Tensor,         # (B, 2)
    c: torch.Tensor,         # (B, n)
    transit: torch.Tensor,   # (B, n)
    decay: torch.Tensor,     # (B,)
    epsilon: torch.Tensor,   # (B,)
    Hz: torch.Tensor,        # (F,)
) -> torch.Tensor:
    """
    Compute cross spectral density (CSD) analytically for a batch of systems.
    Returns complex tensor (B, F, n, n), where B is the batch size, F is the
    number of frequencies, and n is the number of regions.
    """


    A = A.float()
    a = a.float()
    b = b.float()
    c = c.float()
    transit = transit.float()
    decay = decay.float()
    epsilon = epsilon.float()
    Hz = Hz.float()


    B, n, _ = A.shape
    Hz = Hz.reshape(-1)
    F = Hz.shape[0]

    device = A.device
    real_dtype = A.dtype
    cplx_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    # Angular frequency
    w = 2.0 * torch.pi * Hz.to(real_dtype).to(device)  # (F,)

    # Transform log-scaled parameters
    a1 = torch.exp(a[:, 0])                 # (B,)
    a2 = torch.exp(a[:, 1])                 # (B,)
    b1 = torch.exp(b[:, 0])                 # (B,)
    b2 = torch.exp(b[:, 1]) / 2.0           # (B,)
    c_vec = torch.exp(c)                    # (B, n)
    tau_vec = 2.0 * torch.exp(transit)      # (B, n)
    k = 0.64 * torch.exp(decay.squeeze(-1))     # (B,)
    eps = 1.0 * torch.exp(epsilon.squeeze(-1))  # (B,)

    # Reverse log-scaling of diagonal of A
    A_eff = A.clone()
    idx = torch.arange(n, device=device)
    A_eff[:, idx, idx] = -0.5 * torch.exp(A_eff[:, idx, idx])

    # Fixed hemodynamic parameters
    gam = 0.32
    alph = 0.32
    TE = 0.04
    V0 = 4.0
    r0 = 25.0
    nu0 = 40.3
    E0 = 0.4

    # Power spectra
    w_pow = w[None, :]                           # (1, F)

    # endogenous fluctuations
    Gu_un = w_pow.pow(-a2[:, None])              # (B, F)
    Gu = a1[:, None] * Gu_un / (Gu_un.sum(dim=1, keepdim=True) + 1e-15)  # (B, F)

    # observation noise (global)
    Gn2_un = w_pow.pow(-b2[:, None])             # (B, F)
    Gn2 = b1[:, None] * Gn2_un / (Gn2_un.sum(dim=1, keepdim=True) + 1e-15)  # (B, F)

    # observation noise (region-specific)
    G_un = w_pow.pow(-b2[:, None])               # (B, F)
    G_norm = G_un / (G_un.sum(dim=1, keepdim=True) + 1e-15)  # (B, F)
    Cdiag = torch.diag_embed(c_vec)              # (B, n, n)
    Gn1 = G_norm[:, :, None, None] * Cdiag[:, None, :, :]    # (B, F, n, n)

    # HRF(w, k, tau, eps) vectorized
    wBFn = w[None, :, None]
    iwBFn = (1j * wBFn).to(cplx_dtype)           # (1, F, 1)
    tauBFn = tau_vec[:, None, :]                 # (B, F, n)
    kBF1 = k[:, None, None]                      # (B, 1, 1)
    epsBF1 = eps[:, None, None]                  # (B, 1, 1)

    const1 = -epsBF1 + E0 * TE * nu0 * 4.3 + 1.0   
    const2 = (nu0 * 43.0 + epsBF1 * r0 * 10.0)     
    log1mE0 = torch.log(torch.tensor(1.0 - E0, dtype=real_dtype, device=device))

    num = V0 * (
        E0 * alph * (tauBFn * iwBFn + 1.0) * const1
        - (E0 * TE * log1mE0 * (alph * tauBFn * iwBFn + 1.0) * const2 * (E0 - 1.0)) / 10.0
    ) * (-1.0 / 16.0)

    den = (
        E0
        * (alph * tauBFn * iwBFn + 1.0)
        * (tauBFn * iwBFn + 1.0)
        * (gam + kBF1 * iwBFn - (wBFn ** 2))
    )

    Hvals = (num / den).to(cplx_dtype)      # (B, F, n)
    H = torch.diag_embed(Hvals)             # (B, F, n, n)
    Hh = torch.diag_embed(Hvals.conj())     # (B, F, n, n)

    I = torch.eye(n, dtype=cplx_dtype, device=device)
    A_c = A_eff.to(cplx_dtype)

    iw = (1j * w).to(cplx_dtype).reshape(1, -1, 1, 1) # (1, F, 1, 1)
    # M = iw*I - A_eff
    M = iw * I[None, None, :, :] - A_c[:, None, :, :]
    # SAFETY ADDITION: Add a tiny "jitter" to the diagonal of M
    # This prevents the matrix from being perfectly singular
    jitter = 1e-6 * I[None, None, :, :]
    M = M + jitter

    # M = iwBFn[:, :, None] * I[None, None, :, :] - A_c[:, None, :, :]  # (B, F, n, n)
    A1 = torch.linalg.inv(M)  # (B, F, n, n)

    # S = A1 * A1'
    S = A1 @ A1.conj().transpose(-1, -2)    # (B, F, n, n)
    HAS = H @ S @ Hh                        # (B, F, n, n)

    CSD = (
        Gu[:, :, None, None].to(cplx_dtype) * HAS
        + Gn1.to(cplx_dtype)
        + Gn2[:, :, None, None].to(cplx_dtype)
    )
    return CSD


def load_npz_array(path: str, key: Optional[str] = None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path)
        if key is None:
            if len(d.files) != 1:
                raise ValueError(f"{path} contains multiple arrays {d.files}. Specify key.")
            key = d.files[0]
        if key not in d:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {d.files}")
        return d[key]
    elif ext == ".npy":
        return np.load(path)
    elif ext in (".txt", ".csv"):
        return np.loadtxt(path, delimiter="," if ext == ".csv" else None)
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load NPZ parameters, compute CSD with PyTorch, and compare to saved CSD."
    )
    parser.add_argument("--dir", type=str, default=".",
                        help="Directory with *.npz files")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (e.g., cpu, cuda, cuda:0)")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for comparison")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for comparison")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                        help="Computation dtype")
    args = parser.parse_args()

    d = args.dir

    # Load arrays
    A_np        = load_npz_array(os.path.join(d, "A.npz"),        key="A")
    transit_np  = load_npz_array(os.path.join(d, "transit.npz"),  key="transit")
    decay_np    = load_npz_array(os.path.join(d, "decay.npz"),    key="decay")
    epsilon_np  = load_npz_array(os.path.join(d, "epsilon.npz"),  key="epsilon")
    a_np        = load_npz_array(os.path.join(d, "aa.npz"),        key="a")
    b_np        = load_npz_array(os.path.join(d, "b.npz"),        key="b")
    c_np        = load_npz_array(os.path.join(d, "c.npz"),        key="c")
    CSD_ref_np  = load_npz_array(os.path.join(d, "CSD.npz"),      key="CSD")
    Hz_np       = load_npz_array(os.path.join(d, "Hz.npz"),       key="Hz")

    B, n, _ = A_np.shape

    # Filter NaNs
    valid_A = ~np.isnan(A_np).reshape(B, -1).any(axis=1)
    valid_CSD = ~np.isnan(CSD_ref_np).reshape(B, -1).any(axis=1)
    valid_idx = valid_A & valid_CSD

    if not valid_idx.all():
        num_invalid = B - valid_idx.sum()
        print(f"Excluding {num_invalid} batches containing NaNs...")
        A_np        = A_np[valid_idx]
        transit_np  = transit_np[valid_idx]
        decay_np    = decay_np[valid_idx]
        epsilon_np  = epsilon_np[valid_idx]
        a_np        = a_np[valid_idx]
        b_np        = b_np[valid_idx]
        c_np        = c_np[valid_idx]
        CSD_ref_np  = CSD_ref_np[valid_idx]
        B = A_np.shape[0] 

    F = Hz_np.shape[0]

    # Torch dtype
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    cplx_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    # Convert to torch tensors
    A_t        = torch.from_numpy(A_np).to(dtype).to(args.device)
    a_t        = torch.from_numpy(a_np).to(dtype).to(args.device)
    b_t        = torch.from_numpy(b_np).to(dtype).to(args.device)
    c_t        = torch.from_numpy(c_np).to(dtype).to(args.device)
    transit_t  = torch.from_numpy(transit_np).to(dtype).to(args.device)
    decay_t    = torch.from_numpy(decay_np).reshape(B, 1).to(dtype).to(args.device)
    epsilon_t  = torch.from_numpy(epsilon_np).reshape(B, 1).to(dtype).to(args.device)
    Hz_t       = torch.from_numpy(Hz_np).to(dtype).to(args.device)
    
    # FORCE strict dimension constraints to prevent silent 5D broadcasting
    a_t        = a_t.view(B, 2)
    b_t        = b_t.view(B, 2)
    c_t        = c_t.view(B, n)
    transit_t  = transit_t.view(B, n)

    # Compute CSD in torch
    with torch.no_grad():
        CSD_torch = spm_csd_analytic_torch(
            A=A_t, a=a_t, b=b_t, c=c_t,
            transit=transit_t, decay=decay_t, epsilon=epsilon_t,
            Hz=Hz_t
        )

    # Compare to reference
    CSD_ref = torch.from_numpy(CSD_ref_np).to(cplx_dtype).to(args.device)

    exact_equal = torch.equal(CSD_torch, CSD_ref)
    abs_err = (CSD_torch - CSD_ref).abs()
    rel_err = abs_err / (CSD_ref.abs().clamp_min(1e-20))
    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()

    allclose = torch.allclose(CSD_torch, CSD_ref, atol=args.atol, rtol=args.rtol)

    print("=== CSD Torch vs Reference Report ===")
    print(f"Batch size (B): {B}, Nodes (n): {n}, Frequencies (F): {F}")
    print(f"Device: {args.device}, Dtype: {args.dtype}")
    print(f"Exact equality: {exact_equal}")
    print(f"Allclose (rtol={args.rtol}, atol={args.atol}): {allclose}")
    print(f"Max abs error: {max_abs:.6g}")
    print(f"Max rel error: {max_rel:.6g}")

    if exact_equal or allclose:
        print("SUCCESS: CSD_torch matches the saved CSD within tolerance.")
        sys.exit(0)
    else:
        print("FAIL: CSD_torch does not match the saved CSD within tolerance.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()