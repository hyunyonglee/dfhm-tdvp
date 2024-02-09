import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tebd
from tenpy.tools.process import mkl_set_nthreads
import argparse
import logging.config
import os
import os.path
import h5py
from tenpy.tools import hdf5_io
import model

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def measurements(psi, L):
    
    # Measurements
    EE = psi.entanglement_entropy()
    Nu = psi.expectation_value("Nu")
    Nd = psi.expectation_value("Nd")
    
    # measuring correlation functions

    I0 = int(L/3)
    R_CORR = int(L/3)
    
    cor_dipole_uu = []
    cor_dipole_dd = []
    cor_dipole_ud = []

    for i in range(R_CORR):
        cor_dipole_uu.append( np.abs( psi.expectation_value_term([('Cdu',I0+1),('Cu',I0),('Cdu',I0+2+i),('Cu',I0+3+i)]) ) )
        cor_dipole_dd.append( np.abs( psi.expectation_value_term([('Cdd',I0+1),('Cd',I0),('Cdd',I0+2+i),('Cd',I0+3+i)]) ) )
        cor_dipole_ud.append( np.abs( psi.expectation_value_term([('Cdu',I0+1),('Cu',I0),('Cdd',I0+2+i),('Cd',I0+3+i)]) ) )

    return EE, Nu, Nd, cor_dipole_uu, cor_dipole_dd, cor_dipole_ud


def write_data( psi, E, EE, Nu, Nd, cor_dipole_uu, cor_dipole_dd, cor_dipole_ud, L, t, tp, U, path ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    data = {"psi": psi}
    with h5py.File(path+"/mps/psi_L_%d_t_%.2f_tp_%.2f_U_%.2f.h5" % (L,t,tp,U), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Nus = open(path+"/observables/Nus.txt","a", 1)
    file_Nds = open(path+"/observables/Nds.txt","a", 1)
    
    file_EE.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + "  ".join(map(str, EE)) + " " + "\n")
    file_Nus.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + "  ".join(map(str, Nu)) + " " + "\n")
    file_Nds.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + "  ".join(map(str, Nd)) + " " + "\n")
    
    file_EE.close()
    file_Nus.close()
    file_Nds.close()
    
    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + repr(E) + " " + repr(np.max(EE)) + " " + "\n")
    file.close()

    file_corr_uu = open(path+"/observables/corr_dipole_uu.txt","a", 1)
    file_corr_dd = open(path+"/observables/corr_dipole_dd.txt","a", 1)
    file_corr_ud = open(path+"/observables/corr_dipole_ud.txt","a", 1)

    file_corr_uu.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + "  ".join(map(str, cor_dipole_uu)) + " " + "\n")
    file_corr_dd.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + "  ".join(map(str, cor_dipole_dd)) + " " + "\n")
    file_corr_ud.write(repr(t) + " " + repr(tp) + " " + repr(U) + " " + "  ".join(map(str, cor_dipole_ud)) + " " + "\n")
    

if __name__ == "__main__":
    
    current_directory = os.getcwd()
    conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
    }
    logging.config.dictConfig(conf)

    parser=argparse.ArgumentParser()
    parser.add_argument("--L", default='10', help="Length of chain")
    parser.add_argument("--t", default='1.0', help="3-site Dipolar hopping amplitude")
    parser.add_argument("--tp", default='1.0', help="4-site Dipolar hopping amplitude")
    parser.add_argument("--U", default='20.0', help="On-site Hubbard interaction")
    parser.add_argument("--chi", default='64', help="Bond dimension")
    parser.add_argument("--init_state", default='half-filled-spin-zero', help="Initial state")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    parser.add_argument("--RM", default=None, help="random initial state")
    parser.add_argument("--max_sweep", default='100', help="Maximum number of sweeps")
    args=parser.parse_args()

    L = int(args.L)
    t = float(args.t)
    tp = float(args.tp)
    U = float(args.U)
    chi = int(args.chi)
    RM = args.RM
    init_state = args.init_state
    path = args.path
    max_sweep = int(args.max_sweep)
    
    model_params = {
    "L": L, 
    "t": t,
    "tp": tp,
    "U": U,
    }

    DFHM = model.DIPOLAR_FERMI_HUBBARD_CONSERVED(model_params)

    # initial state
    if init_state == 'half-filled-spin-zero':
        product_state = ['up','down'] * int(DFHM.lat.N_sites/2)

    psi = MPS.from_product_state(DFHM.lat.mps_sites(), product_state, bc=DFHM.lat.bc_MPS)

    if RM == 'random':
        TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 50}, 'verbose': 0}
        eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
        eng.run()
        psi.canonical_form() 

    chi_list = {}
    dchi = int((chi - 50)/10)
    for i in range(11):
        chi_list[5*i] = 50 + i*dchi
    
    dmrg_params = {
    # 'mixer': True,  # setting this to True helps to escape local minima
        'mixer' : dmrg.SubspaceExpansion,
        'mixer_params': {
            'amplitude': 1.e-2,
            'decay': 1.5,
            'disable_after': 20
        },
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-9
        },
        'chi_list': chi_list,
        'max_E_err': 1.0e-9,
        'max_S_err': 1.0e-9,
        'max_sweeps': max_sweep,
        'combine' : True,
        'diag_method': 'lanczos',
        'lanczos_params': {
            'N_max': 3,  # fix the number of Lanczos iterations: the number of `matvec` calls
            'N_min': 3,
            'N_cache': 20,  # keep the states during Lanczos in memory
            'reortho': False
        }
    }

    # ground state
    eng = dmrg.TwoSiteDMRGEngine(psi, DFHM, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    psi.canonical_form() 

    EE, Nu, Nd, cor_dipole_uu, cor_dipole_dd, cor_dipole_ud = measurements(psi, L)
    write_data( psi, E, EE, Nu, Nd, cor_dipole_uu, cor_dipole_dd, cor_dipole_ud, L, t, tp, U, path )