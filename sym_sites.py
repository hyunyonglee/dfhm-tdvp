# Copyright 2023 Hyun-Yong Lee

import numpy as np
from tenpy.networks.site import Site
from tenpy.linalg import np_conserved as npc


def BosonSite_DM_conserved(Nmax=2, cons_N='N', cons_D='D', x=0):
    
    dim = Nmax + 1
    states = [str(n) for n in range(0, dim)]
    if dim < 2:
       raise ValueError("local dimension should be larger than 1....")
    
    # 0) define the operators
    B = np.zeros([dim, dim], dtype=np.float64)  # destruction/annihilation operator
    for n in range(1, dim):
       B[n - 1, n] = np.sqrt(n)
    Bd = np.transpose(B)  # .conj() wouldn't do anything
    # Note: np.dot(Bd, B) has numerical roundoff errors of eps~=4.4e-16.
    Ndiag = np.arange(dim, dtype=np.float64)
    N = np.diag(Ndiag)
    NN = np.diag(Ndiag**2)
    dN = np.diag(Ndiag)
    dNdN = np.diag((Ndiag)**2)
    P = np.diag(1. - 2. * np.mod(Ndiag, 2))
    ops = dict(B=B, Bd=Bd, N=N, NN=NN, dN=dN, dNdN=dNdN, P=P)

    # 1) handle charges
    qmod = []
    qnames = []
    charges = []
    if cons_N == 'N':
        qnames.append('N')
        qmod.append(1)
        charges.append( [i for i in range(dim)] )

    if cons_D == 'D':
        qnames.append('D')  # Dipole moment
        qmod.append(1)
        charges.append( [i*x for i in range(dim)] )

    if len(qmod) == 1:
        charges = charges[0]
    else:  # len(charges) == 2: need to transpose
        charges = [[q1, q2] for q1, q2 in zip(charges[0], charges[1])]
    chinfo = npc.ChargeInfo(qmod, qnames)
    leg = npc.LegCharge.from_qflat(chinfo, charges)

    return Site(leg, states, sort_charge=True, **ops)


def SpinHalfFermionSite_DM_conserved(cons_N='N', cons_Sz='Sz', cons_D='D', x=0):
    
    d = 4
    states = ['empty', 'up', 'down', 'full']
    # 0) Build the operators.
    Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float64)
    Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float64)
    Nu = np.diag(Nu_diag)
    Nd = np.diag(Nd_diag)
    Ntot = np.diag(Nu_diag + Nd_diag)
    dN = np.diag(Nu_diag + Nd_diag)
    NuNd = np.diag(Nu_diag * Nd_diag)
    JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
    JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
    JW = JWu * JWd  # (-1)^{Nu+Nd}

    Cu = np.zeros((d, d))
    Cu[0, 1] = Cu[2, 3] = 1
    Cdu = np.transpose(Cu)
    # For spin-down annihilation operator: include a Jordan-Wigner string JWu
    # this ensures that Cdu.Cd = - Cd.Cdu
    # c.f. the chapter on the Jordan-Wigner trafo in the userguide
    Cd_noJW = np.zeros((d, d))
    Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
    Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
    Cdd = np.transpose(Cd)

    # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
    # where S^gamma is the 2x2 matrix for spin-half
    Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
    Sp = np.dot(Cdu, Cd)
    Sm = np.dot(Cdd, Cu)
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)

    ops = dict(JW=JW, JWu=JWu, JWd=JWd,
               Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
               Nu=Nu, Nd=Nd, Ntot=Ntot, NuNd=NuNd, dN=dN,
               Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable

    # handle charges
    qmod = []
    qnames = []
    charges = []
    if cons_N == 'N':
        qnames.append('N')
        qmod.append(1)
        charges.append([0, 1, 1, 2])
    elif cons_N == 'parity':
        qnames.append('parity_N')
        qmod.append(2)
        charges.append([0, 1, 1, 0])
    if cons_Sz == 'Sz':
        qnames.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
        qmod.append(1)
        charges.append([0, 1, -1, 0])
        del ops['Sx']
        del ops['Sy']
    elif cons_Sz == 'parity':
        qnames.append('parity_Sz')  # the charge is (2*Sz) mod (2*2)
        qmod.append(4)
        charges.append([0, 1, 3, 0])  # == [0, 1, -1, 0] mod 4
        # e.g. terms like `Sp_i Sp_j + hc` with Sp=Cdu Cd have charges 'N', 'parity_Sz'.
        # The `parity_Sz` is non-trivial in this case!
    
    if cons_D == 'D':
        qnames.append('D')  # Dipole moment
        qmod.append(1)
        charges.append( [0, x, x, 2*x] )

    if len(qmod) == 0:
        leg = npc.LegCharge.from_trivial(d)
    else:
        if len(qmod) == 1:
            charges = charges[0]
        else:  # len(charges) == 2: need to transpose
            charges = [[q1, q2, q3] for q1, q2, q3 in zip(charges[0], charges[1], charges[2])]
        chinfo = npc.ChargeInfo(qmod, qnames)
        leg = npc.LegCharge.from_qflat(chinfo, charges)
        
    site = Site(leg, states, sort_charge=True, **ops)
    site.need_JW_string = set(['Cu', 'Cdu', 'Cd', 'Cdd', 'JWu', 'JWd', 'JW'])
    # site.need_JW_string |= set(['Cu', 'Cdu', 'Cd', 'Cdd'])
    return site
