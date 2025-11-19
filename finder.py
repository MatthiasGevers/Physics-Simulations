#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find DARK states for the collective-lowering dissipator built w.r.t. H_B.
"""

import math
import numpy as np
from typing import Dict, List, Tuple

from qutip import (
    Qobj, tensor, qeye, basis,
    sigmaz, sigmam, expect
)

# ============================ CONFIG ============================
N: int          = 4       # number of spins
h: float        = 1.0      # uniform field in H_B
tol: float      = 1e-10    # tolerance to detect zero eigenvalues of K
max_terms: int  = 8        # how many largest components to print per dark vector
full: bool      = False    # if True, print full component list

# Printing controls for the max-ergotropy combination (coeffs in dark basis)
combo_max_terms: int = 16  # show this many largest coefficients
combo_tol: float     = 1e-8  # hide coefficients with |c| below this
# ===============================================================

# ---------- utilities ----------

def op_on_site(op: Qobj, i: int, N: int) -> Qobj:
    """Place single-qubit operator `op` on site i (0-based) in an N-qubit tensor product."""
    ops = [qeye(2)] * N
    ops[i] = op
    return tensor(ops)

def build_HB(N: int, h: float) -> Qobj:
    """H_B = (h/2) * Σ σ^z_i."""
    return 0.5 * h * sum(op_on_site(sigmaz(), i, N) for i in range(N))

def build_J_minus(N: int) -> Qobj:
    """Collective lowering J_- = Σ σ^-_i."""
    return sum(op_on_site(sigmam(), i, N) for i in range(N))

def bitstring(idx: int, N: int) -> str:
    return format(idx, f"0{N}b")

def computational_ket(N: int, idx: int) -> Qobj:
    """|idx> with dims as an N-qubit tensor product (keeps dims consistent)."""
    bits = bitstring(idx, N)
    kets = [basis(2, int(b)) for b in bits]
    return tensor(kets)

def group_indices_by_energy_HB(N: int, h: float) -> Dict[float, List[int]]:
    """
    For uniform h, the energy depends only on number of '1' (down) bits.
    Returns mapping E -> [indices with that energy], using computational basis.
    """
    groups: Dict[float, List[int]] = {}
    dim = 2**N
    for idx in range(dim):
        b = bitstring(idx, N)
        n_down = b.count("1")
        E = 0.5 * h * (N - 2 * n_down)  # E = (h/2) * sum z_i = (h/2)*(#0 - #1)
        groups.setdefault(E, []).append(idx)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))

def projector_from_indices(N: int, indices: List[int]) -> Qobj:
    """Π = Σ_{i in indices} |i⟩⟨i| in the N-qubit space (dense but fine for N<=10)."""
    dim = 2**N
    data = np.zeros((dim, dim), dtype=complex)
    for i in indices:
        data[i, i] = 1.0
    dims = [[2]*N, [2]*N]
    return Qobj(data, dims=dims)

def bohr_decompose_wrt_HB(N: int, h: float, A: Qobj, tol: float = 1e-12) -> Dict[float, Qobj]:
    """
    Construct X(ω) blocks for operator A (here A=J_-) w.r.t. H_B.
    Returns dict ω>0 -> X(ω) = Σ_{E'-E=ω} Π_E A Π_{E'}.
    """
    E_groups = group_indices_by_energy_HB(N, h)  # E -> list of basis indices at that energy
    Es = list(E_groups.keys())
    P: Dict[float, Qobj] = {E: projector_from_indices(N, idxs) for E, idxs in E_groups.items()}

    blocks: Dict[float, Qobj] = {}
    for Ei in Es:
        for Ej in Es:
            w = Ej - Ei
            if w <= tol:
                continue  # only keep ω>0 (downward transitions at T=0)
            X_ij = P[Ei] * A * P[Ej]
            if X_ij.norm() < tol:
                continue
            if w in blocks:
                blocks[w] = blocks[w] + X_ij
            else:
                blocks[w] = X_ij

    blocks = {float(w): X for w, X in blocks.items() if X.norm() > tol}
    return dict(sorted(blocks.items(), key=lambda kv: kv[0]))

def ergotropy_pure(H: Qobj, psi: Qobj) -> float:
    """
    Ergotropy for a pure state: W = <H> - E_min(H).
    (The passive state for a rank-1 ρ puts the lone eigenvalue 1 on the ground state.)
    """
    E_min = float(min(H.eigenenergies()))
    E_exp = float(expect(H, psi))
    return E_exp - E_min

def print_state_components(psi: Qobj, N: int, max_terms: int = 8, tol: float = 1e-8, full: bool = False):
    """Print components of |psi> in computational basis, sorted by |amplitude|."""
    vec = np.array(psi.full()).reshape(-1)
    items = []
    for idx, amp in enumerate(vec):
        mag = abs(amp)
        if mag > tol:
            items.append((mag, idx, amp))
    items.sort(reverse=True, key=lambda t: t[0])

    if not items:
        print("    (all components below threshold)")
        return

    if not full:
        items = items[:max_terms]

    for mag, idx, amp in items:
        bs = bitstring(idx, N)
        print(f"    |{bs}> : {amp.real:+.6f}{amp.imag:+.6f}i   (|amp|={mag:.6f})")

# ---------- degeneracy helpers ----------

def top_degeneracy_spectrum(H_dark: np.ndarray, w: np.ndarray, rel_tol: float = 1e-12) -> int:
    """Count eigenvalues equal to λ_max within a relative tolerance."""
    lam_max = float(w[-1])
    denom = max(1.0, abs(lam_max))
    return int(np.sum(np.abs(w - lam_max) <= rel_tol * denom))

def top_degeneracy_svd(H_dark: np.ndarray, w: np.ndarray, rel_tol: float = 1e-12) -> int:
    """Nullity of (H_dark - λ_max I) via SVD (more robust with clustered roots)."""
    lam_max = float(w[-1])
    M = H_dark - lam_max * np.eye(H_dark.shape[0])
    svals = np.linalg.svd(M, compute_uv=False)
    # standard perturbation-aware tolerance
    tol = rel_tol * max(H_dark.shape) * max(1.0, np.max(np.abs(w)))
    return int(np.sum(svals <= tol))

def choose_kstar_down(N: int) -> int:
    """For max ergotropy (max N_up), the corresponding number of downs is k* = ceil(N/2)."""
    return (N + 1) // 2  # ceil(N/2)

def deg_analytic(N: int) -> int:
    """
    Closed-form degeneracy of the top-ergotropy level inside the dark space
    for collective J_- built w.r.t H_B:
        deg_max = C(N, ceil(N/2)) - C(N, ceil(N/2)+1)
    (with C(N, k)=0 outside range).
    """
    k = choose_kstar_down(N)
    c = lambda n, r: math.comb(n, r) if 0 <= r <= n else 0
    return c(N, k) - c(N, k + 1)

def intersection_deg(B_mat: np.ndarray, N: int, tol: float = 1e-10) -> int:
    """
    Compute deg = dim( span(B) ∩ H_{k*} ), where B_mat's columns span the dark space,
    and H_{k*} is the Hamming layer with exactly k* = ceil(N/2) '1's (downs).
    """
    kstar = choose_kstar_down(N)
    idxs = [idx for idx in range(2**N) if bitstring(idx, N).count("1") == kstar]
    if len(idxs) == 0:
        return 0
    C_cols = [computational_ket(N, i).full().reshape(-1) for i in idxs]
    C_mat = np.column_stack(C_cols)                  # (2^N) x dim(H_k*)
    # Orthonormalize both subspace bases (QR)
    Q_B, _ = np.linalg.qr(B_mat)
    Q_C, _ = np.linalg.qr(C_mat)
    O = Q_B.conj().T @ Q_C                           # overlap matrix
    s = np.linalg.svd(O, compute_uv=False)
    return int(np.sum(s > tol))                      # principal angles == 0 ⇒ common dim

# ---------- main analysis ----------

def main():
    print("="*70)
    print("Collective-lowering DARK-state analysis (dissipator built w.r.t. H_B)")
    print("="*70)
    print(f"N={N}, h={h:.6f}\n")

    # Build H_B and J_-
    H_B = build_HB(N, h)
    Jm  = build_J_minus(N)

    # Bohr decomposition of J_- w.r.t. H_B
    blocks = bohr_decompose_wrt_HB(N, h, Jm, tol=1e-12)
    if not blocks:
        print("[Bohr(H_B)] No ω>0 blocks found (unexpected). Exiting.")
        return

    omegas = sorted(blocks.keys())
    print(f"[Bohr(H_B)] ω>0 set: {[f'{w:.6f}' for w in omegas]}")
    if len(omegas) == 1 and abs(omegas[0] - h) < 1e-9:
        diff_norm = (blocks[omegas[0]] - Jm).norm()
        print(f"[check] || X(ω≈h) - J_- || = {diff_norm:.3e}")

    # Build collapse operators (rates don't affect the kernel; set γ=1)
    Ls: List[Qobj] = [blocks[w] for w in omegas]

    # Build K = Σ L†L
    K = sum(L.dag() * L for L in Ls)

    # Diagonalize K; dark space = eigenspace at λ≈0
    evals, evecs = K.eigenstates()
    evals = np.array([float(ev) for ev in evals])
    dark_idx = [i for i, lam in enumerate(evals) if abs(lam) < tol]
    dim_dark = len(dark_idx)

    print(f"[K] spectrum min..max = {evals.min():.3e} .. {evals.max():.3e}")
    print(f"[DFS(H_B, collective)] dim ker(K) = {dim_dark}")

    # Theoretical dimension for uniform h and collective lowering: C(N, ⌊N/2⌋)
    try:
        theo = math.comb(N, N//2)
        print(f"[theory] dim ker(J_-) = C(N, ⌊N/2⌋) = {theo}")
    except Exception:
        pass

    # Ground energy for ergotropy
    E0 = float(min(H_B.eigenenergies()))
    print(f"[H_B] ground energy E_min = {E0:.6f}\n")

    if dim_dark == 0:
        print("No dark states found at the chosen tolerance.")
        return

    # Collect dark kets and their ergotropies
    dark_kets: List[Qobj] = [evecs[i] for i in dark_idx]
    dark_with_W: List[Tuple[float, Qobj]] = [(ergotropy_pure(H_B, psi), psi) for psi in dark_kets]

    # Sort ascending by ergotropy so the biggest one prints last
    dark_with_W.sort(key=lambda t: t[0])

    # Report in sorted order
    for j, (W, psi) in enumerate(dark_with_W, start=1):
        annih_norm = float((Jm * psi).norm())
        print(f"--- Dark vector #{j} ---")
        print(f"Ergotropy W(H_B) = {W:.9f}")
        print(f"|| J_- |ψ> || = {annih_norm:.3e} (should be ~0)")
        print("Components in computational basis (0≡|0⟩ up, 1≡|1⟩ down):")
        print_state_components(psi, N, max_terms=max_terms, tol=1e-8, full=full)
        print()

    # Optional: show that P_dark commutes with H_B (often ~0 here)
    P_dark = sum(psi * psi.dag() for _, psi in dark_with_W)
    comm = P_dark * H_B - H_B * P_dark
    comm_norm_fro = float(np.sqrt((comm.dag() * comm).tr().real))
    print(f"[Invariant?] ||[P_dark, H_B]||_F = {comm_norm_fro:.3e}  (~0 ⇒ H_B blocks the dark space)\n")

    # ===== Max-ergotropy dark superposition (optimize inside DFS) =====
    print("="*70)
    print("Max-ergotropy dark superposition (within the printed dark basis)")
    print("="*70)

    # Build an isometry B from the *printed order* (ascending W)
    ordered_dark_kets = [psi for (_, psi) in dark_with_W]
    B_cols = [psi.full().reshape(-1) for psi in ordered_dark_kets]   # columns are dark basis vectors
    B_mat  = np.column_stack(B_cols)                                  # shape: (2**N, dim_dark)
    B = Qobj(B_mat, dims=[[2]*N, [dim_dark]])

    # Project H_B into the dark space and diagonalize the small matrix
    H_dark = (B.dag() * H_B * B).full()                              # dim_dark x dim_dark (numpy array)
    w, V = np.linalg.eigh(H_dark)                                    # Hermitian => real eigs
    lam_max = float(w[-1])
    v_max = V[:, -1]                                                 # coefficients in the printed dark basis

    # Lift the maximizer back to full Hilbert space
    psi_max = (B * Qobj(v_max, dims=[[dim_dark], [1]])).unit()

    # Report metrics
    W_max = ergotropy_pure(H_B, psi_max)
    annih_resid = float((Jm * psi_max).norm())
    Nz = sum(op_on_site(sigmaz(), i, N) for i in range(N))
    n_up = 0.5 * (N + float(expect(Nz, psi_max)))                    # ⟨# of |0⟩ (up) spins⟩

    print(f"[OPT] Max ergotropy W_max(H_B) = {W_max:.12f}   (theory ≈ {h * (N//2):.12f})")
    print(f"[OPT] || J_- |ψ_max> || = {annih_resid:.3e}     (should be ~0)")
    print(f"[OPT] ⟨N_up⟩ ≈ {n_up:.12f}                   (should be ≈ {N//2})")

    # Print the combination in terms of the printed dark vectors
    coeffs = list(enumerate(v_max, start=1))  # (index in printed order, complex coeff)
    coeffs.sort(key=lambda t: abs(t[1]), reverse=True)
    # Filter and truncate for readability
    coeffs = [(j, c) for (j, c) in coeffs if abs(c) > combo_tol]
    if not coeffs:
        print("\n[OPT] All combination coefficients are below the display threshold.")
    else:
        print("\n[OPT] Combination coefficients in the printed dark basis (ascending W):")
        for (idx, c) in coeffs[:combo_max_terms]:
            print(f"  c_{idx:>4d} = {c.real:+.8f}{c.imag:+.8f}i   (|c|={abs(c):.8f})")

    # Optionally show the leading computational-basis components of ψ_max
    print("\n[OPT] Leading components of the max-ergotropy dark state |ψ_max⟩:")
    print_state_components(psi_max, N, max_terms=max_terms, tol=1e-8, full=False)

    # ===== New: Degeneracy of the max-ergotropy level =====
    print("\n" + "="*70)
    print("Degeneracy of the max-ergotropy dark level inside the DFS")
    print("="*70)

    # Numerical degeneracy from H_dark
    deg_spec = top_degeneracy_spectrum(H_dark, w, rel_tol=1e-12)
    deg_svd  = top_degeneracy_svd(H_dark, w, rel_tol=1e-12)
    print(f"[DEG-num] multiplicity from eigvals(H_dark): {deg_spec}")
    print(f"[DEG-num] multiplicity via SVD nullity(H_dark - λ_max I): {deg_svd}")

    # Numerical degeneracy via intersection with Hamming layer k* = ceil(N/2) downs
    deg_intr = intersection_deg(B_mat, N, tol=1e-10)
    print(f"[DEG-num] dim(DFS ∩ Hamming layer k*={choose_kstar_down(N)} downs): {deg_intr}")

    # Analytic closed form (collective J_- model)
    deg_an = deg_analytic(N)
    print(f"[DEG-ana] predicted degeneracy = C(N, ceil(N/2)) - C(N, ceil(N/2)+1) = {deg_an}")

if __name__ == "__main__":
    main()
