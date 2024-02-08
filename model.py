# Copyright 2023 Hyun-Yong Lee

from tenpy.models.lattice import Chain, Lattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.networks.site import BosonSite
import numpy as np
import sym_sites
__all__ = ['DIPOLAR_BOSE_HUBBARD']


class DIPOLAR_FERMI_HUBBARD(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_FERMI_HUBBARD")
        L = model_params.get('L', 1)
        t = model_params.get('t', 1.)
        tp = model_params.get('tp', 0.)
        h = model_params.get('h', 0.)
        U = model_params.get('U', 0.)
        mu = model_params.get('mu', 0.)
        bc_MPS = model_params.get('bc_MPS', 'infinite')
        bc = model_params.get('bc', 'periodic')
        QN = model_params.get('QN', 'N')
        QS = model_params.get('QS', 'Sz')

        site = SpinHalfFermionSite( cons_N=QN, cons_Sz=QS, filling=1.0 )
        site.multiply_operators(['Cu','Cd'])
        site.multiply_operators(['Cd','Cu'])
        site.multiply_operators(['Cdd','Cdu'])
        site.multiply_operators(['Cdu','Cdd'])
        
        # MPS boundary condition
        # if bc_MPS == 'finite' and bc == 'periodic':
        #     order = 'folded'
        # else:
        order = 'default'
        
        lat = Chain( L=L, site=site, bc=bc, bc_MPS=bc_MPS, order=order )
        CouplingModel.__init__(self, lat)

        # 2-site hopping
        self.add_coupling( -h, 0, 'Cdu', 0, 'Cu', 1, plus_hc=True)
        self.add_coupling( -h, 0, 'Cdd', 0, 'Cd', 1, plus_hc=True)
        
        # 3-site hopping
        self.add_multi_coupling( -t, [('Cdu', 0, 0), ('Cu Cd', 1, 0), ('Cdd', 2, 0)])
        self.add_multi_coupling( -t, [('Cdd', 0, 0), ('Cd Cu', 1, 0), ('Cdu', 2, 0)])
        
        self.add_multi_coupling( -t, [('Cd', 2, 0), ('Cdd Cdu', 1, 0), ('Cu', 0, 0)])
        self.add_multi_coupling( -t, [('Cu', 2, 0), ('Cdu Cdd', 1, 0), ('Cd', 0, 0)])
        

        # 4-site hopping
        self.add_multi_coupling( -tp, [('Cdu', 0, 0), ('Cu', 1, 0), ('Cd', 2, 0), ('Cdd', 3, 0)])
        self.add_multi_coupling( -tp, [('Cdd', 0, 0), ('Cd', 1, 0), ('Cu', 2, 0), ('Cdu', 3, 0)])

        self.add_multi_coupling( -tp, [('Cd', 3, 0), ('Cdd', 2, 0), ('Cdu', 1, 0), ('Cu', 0, 0)])
        self.add_multi_coupling( -tp, [('Cu', 3, 0), ('Cdu', 2, 0), ('Cdd', 1, 0), ('Cd', 0, 0)])
        

        # Onsite Hubbard Interaction
        self.add_onsite( U, 0, 'NuNd')

        # Chemical potential
        self.add_onsite( -mu, 0, 'Ntot')

        
        MPOModel.__init__(self, lat, self.calc_H_MPO())



class DIPOLAR_FERMI_HUBBARD_CONSERVED(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "DIPOLAR_FERMI_HUBBARD")
        L = model_params.get('L', 1)
        t = model_params.get('t', 1.)
        tp = model_params.get('tp', 0.)
        U = model_params.get('U', 0.)
        
        sites = [ sym_sites.SpinHalfFermionSite_DM_conserved(cons_N='N', cons_Sz='Sz', cons_D='D', x=x) for x in range(L) ]
        for x in range(L):
            sites[x].multiply_operators(['Cu','Cd'])
            sites[x].multiply_operators(['Cd','Cu'])
            sites[x].multiply_operators(['Cdd','Cdu'])
            sites[x].multiply_operators(['Cdu','Cdd'])

        lat = Lattice([1], sites, order='default', bc='open', bc_MPS='finite', basis=[[L]], positions=np.array([range(L)]).transpose())

        CouplingModel.__init__(self, lat)

        # 3-site hopping
        for x in range(L-2):
            self.add_multi_coupling( -t, [('Cdu', 0, x), ('Cu Cd', 0, x+1), ('Cdd', 0, x+2)])
            self.add_multi_coupling( -t, [('Cdd', 0, x), ('Cd Cu', 0, x+1), ('Cdu', 0, x+2)])
        
            self.add_multi_coupling( -t, [('Cd', 0, x+2), ('Cdd Cdu', 0, x+1), ('Cu', 0, x)])
            self.add_multi_coupling( -t, [('Cu', 0, x+2), ('Cdu Cdd', 0, x+1), ('Cd', 0, x)])

        # 4-site hopping
        for x in range(L-3):
            self.add_multi_coupling( -tp, [('Cdu', 0, x), ('Cu', 0, x+1), ('Cd', 0, x+2), ('Cdd', 0, x+3)])
            self.add_multi_coupling( -tp, [('Cdd', 0, x), ('Cd', 0, x+1), ('Cu', 0, x+2), ('Cdu', 0, x+3)])

            self.add_multi_coupling( -tp, [('Cd', 0, x+3), ('Cdd', 0, x+2), ('Cdu', 0, x+1), ('Cu', 0, x)])
            self.add_multi_coupling( -tp, [('Cu', 0, x+3), ('Cdu', 0, x+2), ('Cdd', 0, x+1), ('Cd', 0, x)])
        
        # On-site
        for x in range(L):
            self.add_onsite( U, x, 'NuNd')
            
        MPOModel.__init__(self, lat, self.calc_H_MPO())