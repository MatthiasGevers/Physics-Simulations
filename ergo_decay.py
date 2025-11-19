#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, qeye, sigmaz, sigmam, Qobj, mesolve

# ======================= USER SETTINGS =======================
GAMMA   = 1.0     # dissipation rate γ for L = √γ J_-
T_MAX   = 3.0     # max simulation time
N_T     = 3001    # number of time steps
WITH_H  = False   # True: include H_B in mesolve, False: pure decay (H=0)

# Inits accept bitstrings and equal-weight ± superpositions. All length N.
LISTS = [
    {"name": "N3", "N": 4, "h": 1.0, "inits": ["0000"]},
    {"name": "N3", "N": 4, "h": 1.0, "inits": ["0101+0110-1001-1010"]},
    {"name": "N3", "N": 4, "h": 1.0, "inits": ["0101-0110-1001+1010"]},
    {"name": "N4", "N": 4, "h": 1.0, "inits": ["0011+0101+0110+1001+1010+1100"]},
]
# =============================================================

# ----------------- helpers -----------------
def ket_from_bitstring(bits: str):
    if not bits or any(b not in "01" for b in bits):
        raise ValueError(f"Invalid bitstring '{bits}'. Use only 0/1.")
    return tensor([basis(2, int(b)) for b in bits])

def ket_from_sum(spec: str):
    s = spec.replace(" ", "")
    if s[0] not in "+-":
        s = "+" + s
    terms, sign, buf = [], +1, []
    for ch in s:
        if ch in "+-":
            if buf:
                terms.append((sign, "".join(buf))); buf = []
            sign = +1 if ch == "+" else -1
        else:
            buf.append(ch)
    if buf:
        terms.append((sign, "".join(buf)))
    lengths = {len(bits) for _, bits in terms}
    if len(lengths) != 1:
        raise ValueError("All bitstrings in a superposition must have the same length.")
    psi = 0
    for sgn, bits in terms:
        psi = psi + sgn * ket_from_bitstring(bits)
    return psi.unit()

def parse_init_for_N(entry: str, N: int):
    entry = entry.strip()
    if any(op in entry for op in "+-"):
        psi = ket_from_sum(entry)
        L = len(psi.dims[0])
        if L != N:
            raise ValueError(f"Init '{entry}' has length {L}, but list requires N={N}.")
        return entry, psi
    else:
        if len(entry) != N:
            raise ValueError(f"Bitstring '{entry}' has length {len(entry)}, but list requires N={N}.")
        return entry, ket_from_bitstring(entry)

def build_HB(N: int, h: float) -> Qobj:
    sz_sum = 0
    for i in range(N):
        factors = [qeye(2)] * N
        factors[i] = sigmaz()
        sz_sum = sz_sum + tensor(factors)
    return 0.5 * h * sz_sum

def build_J_minus(N: int) -> Qobj:
    jm = 0
    for i in range(N):
        factors = [qeye(2)] * N
        factors[i] = sigmam()
        jm = jm + tensor(factors)
    return jm

def ergotropy(H: Qobj, rho: Qobj) -> float:
    E_exp = float((rho * H).tr().real)
    eps = np.sort(H.eigenenergies())           # ascending
    r = np.sort(rho.eigenenergies())[::-1]     # descending
    E_passive = float(np.dot(r, eps))
    return max(0.0, E_exp - E_passive)

# ----------------- run -----------------
def run():
    if not LISTS:
        raise ValueError("LISTS is empty. Add at least one experiment list.")

    tlist = np.linspace(0.0, T_MAX, N_T)

    # We'll store traces so we can plot the intensity after the first window closes
    curves_erg = []   # list of (tlist, W_t, label)
    curves_int = []   # list of (tlist, I_t, label)

    for group in LISTS:
        name = group.get("name", "")
        N = int(group["N"]); h = float(group["h"])
        inits = list(group["inits"])

        H_B = build_HB(N, h)
        Jm  = build_J_minus(N)
        Jp  = Jm.dag()
        L   = np.sqrt(GAMMA) * Jm
        H_use = H_B if WITH_H else 0 * H_B

        for spec in inits:
            lab, psi0 = parse_init_for_N(spec, N)
            rho0 = psi0 * psi0.dag()

            # Solve dynamics
            result = mesolve(H=H_use, rho0=rho0, tlist=tlist, c_ops=[L], e_ops=[])
            states = result.states

            # Ergotropy trace
            W_t = [ergotropy(H_B, rho_t) for rho_t in states]

            # Emitted intensity I(t) = γ ⟨J+ J-⟩_t
            JJ = (Jp * Jm)
            I_t = [GAMMA * float((rho_t * JJ).tr().real) for rho_t in states]

            label = f"N={N}, h={h}"
            curves_erg.append((tlist, W_t, label))
            curves_int.append((tlist, I_t, label))

    # -------- Figure 1: Ergotropy --------
    plt.figure(figsize=(10.5, 6.0))
    for t, W, lab in curves_erg:
        plt.plot(t, W, lw=2, label=lab)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel(r"Ergotropy", fontsize=16)
    plt.title(f"Ergotropy decay (γ={GAMMA})", fontsize=18)
    plt.legend(loc='best', frameon=False, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # close this window to get the intensity plot

    # -------- Figure 2: Emitted intensity --------
    plt.figure(figsize=(10.5, 6.0))
    for t, I, lab in curves_int:
        plt.plot(t, I, lw=2, label=lab)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel(r"Emitted intensity", fontsize=16)
    plt.title(f"Emission intensity (γ={GAMMA})", fontsize=18)
    plt.legend(loc='best', frameon=False, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
