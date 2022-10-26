import matscipy
from matscipy import calculators
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch import nn
from torch.functional import F
from copy import copy
from itertools import combinations

from torch.utils.tensorboard import SummaryWriter

def supercell_gen(cell_num1, cell_num2):
    atom_pos1 = cell_num1*2 + 1
    atom_pos2 = cell_num2*2 + 1
    zero_cell = np.zeros(((atom_pos1, atom_pos2)))
    for i in range(atom_pos1):
        for j in range(atom_pos2):
            if i%2 == 1:
                zero_cell[i][j] = abs(j%2)
            elif i%2 == 0:
                zero_cell[i][j] = 1 - abs(j%2)
    return zero_cell

class model(nn.Module):
    def __init__(self, atom_list, param_list):
        super().__init__()
        '''
        atom_list: N*dimension
        param_list: N*number of params per each atom
        per each row:
        r_e, f_e, rho_e, rho_s, alpha, beta, A, B, kappa, lamda,
        f_n0, f_n1, f_n2, f_n3, f_0, f_1, f_2, f_3, eta, f_e
        '''
        #* row1: x-coord, row2: y-coord
        self.weights = nn.Parameter(atom_list)
        self.weights_ = self.weights.clone()
        atom_num = len(atom_list)

        self.ind_inter = torch.combinations(torch.arange(self.weights.size(0)), r=2)
        ind_inter_f = torch.fliplr(self.ind_inter)
        ind_all = torch.cat((self.ind_inter, ind_inter_f), dim=0)
        ind_all = ind_all[ind_all[:, 0].sort()[1]]
        self.ind_block = torch.split(ind_all, atom_num-1)

        self.rho_list = torch.zeros(atom_num)
        self.range = torch.arange(atom_num)
        self.params = param_list

    def f_r(self, r, param):
        '''
        Basis function for f(r), tag 1-2
        Param is nD
        '''
        f_e, beta, r_e, lamda = torch.tensor_split(param, param.size(1), dim=1)
        f_e, beta, r_e, lamda = torch.flatten(f_e), torch.flatten(beta), torch.flatten(r_e), torch.flatten(lamda)

        nume = f_e*torch.exp(-beta*(r/r_e-1))
        deno = 1 + (r/r_e-lamda)**20
        return nume/deno 

    def phi_r(self, r, param):
        ''' 
        Potential for tag 2-1, atomic pair for same species.
        Param is nD.
        '''
        a, b, r_e, alpha, beta, kappa, lamda = torch.tensor_split(param, param.size(1), dim=1)
        a, b = torch.flatten(a), torch.flatten(b)
        r_e, alpha, beta = torch.flatten(r_e), torch.flatten(alpha), torch.flatten(beta)
        kappa, lamda = torch.flatten(kappa), torch.flatten(lamda)

        l_nume = a*torch.exp(-alpha*(r/r_e-1))
        l_deno = 1 + (r/r_e-kappa)**20
        r_nume = b*torch.exp(-beta*(r/r_e-1))
        r_deno = 1 + (r/r_e-lamda)**20

        return l_nume/l_deno - r_nume/r_deno

    def f_rho(self, rho, param):
        ''' 
        Density function for tag 3-2, param is 1D for the centre atom
        '''
        f_n0, f_n1, f_n2, f_n3, f_0, f_1, f_2, f_3, f_e, rho_n, rho_e, rho_0, rho_s, eta = param
        if rho < rho_n:
            return (f_n0 + f_n1*(rho/rho_n-1)
                + f_n2*(rho/rho_n-1)**2 + f_n3*(rho/rho_n-1)**3)
        elif rho_n <= rho < rho_0:
            return (f_0 + f_1*(rho/rho_e-1)
                + f_2*(rho/rho_e-1)**2 + f_3*(rho/rho_e-1)**3)
        else:
            return f_e*(1-torch.log((rho/rho_s)**eta))*(rho/rho_s)**eta

    def forward(self,):
        ''' 
        Calculate the whole energy with keeping gradient
        '''
        coord_inter = self.weights[self.ind_inter]
        param_inter = self.params[self.ind_inter]
        r_res = torch.norm(coord_inter[:,1]-coord_inter[:,0], dim=1)
        #* So this will be the cutoff ?
        effe_r_ind = torch.nonzero(r_res <= 5)
        r_res = r_res[effe_r_ind]
        param_inter = param_inter[effe_r_ind.reshape(-1)]

        fr_0 = self.f_r(r_res, param_inter[:, 0][:, [1, 5, 0, 9]]) #* For f^0(r)
        fr_1 = self.f_r(r_res, param_inter[:, 1][:, [1, 5, 0, 9]]) #* For f^1(r)
        phir_0 = self.phi_r(r_res, param_inter[:, 0][:, [6,7,0,4,5,8,9]]) #* For phi^0(r)
        phir_1 = self.phi_r(r_res, param_inter[:, 1][:, [6,7,0,4,5,8,9]]) #* For phi^1(r)
        phi_01 = fr_1/fr_0*phir_0 + fr_0/fr_1*phir_1 #* For phi^01
        p_e = 1/2*torch.sum(phi_01) #* Potential energy term

        for r_ind in self.range:
            '''
            Extract each atom block containing the centre i and surronding j
            '''
            block = self.ind_block[r_ind]
            coord_block = self.weights[block]
            param_block = self.params[block]
            r_blo = torch.norm(coord_block[:,1]-coord_block[:,0], dim=1)
            rho_i = torch.sum(self.f_r(r_blo, param_block[:, 1][:, [1, 5, 0, 9]])) #* For f^1(r)
            f_rho_ = self.f_rho(rho_i, param_block[:, 0][:, [10,11,12,13,14,15,16,17,19,20,2,21,3,18]][0]) #* For F(rho)

            self.rho_list[r_ind] = f_rho_

        # print(p_e, self.rho_list)
        e_all = p_e + torch.sum(self.rho_list)
        return e_all

def train(model, optimizer, scheduler, path_save, device=None, n = 10000):
    # if device is not None:
    #     iteration = torch.arange(n).to(device)
    # else:
    #     iteration = torch.arange(n)
    # writer = SummaryWriter(log_dir = path_save)
    for i in range(n):
        loss = model.forward()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # if i % 100 == 0:
        #     print(i)
        # writer.add_scalar("Training Loss", loss, i)

    return 

if __name__ == '__main__':
    ''' 
    Generate the primary CoNi cell in planar FCC lattice
    '''
    x_extend, y_extend = 5, 5
    r_equ = 2.7
    cell = supercell_gen(x_extend, y_extend)
    cell_t = np.array(cell, dtype=bool)
    init_weight = np.array([math.sqrt(3)/2*r_equ, 1/2*r_equ])
    coord = np.concatenate([np.where(cell_t)[0].reshape(-1,1),
                            np.where(cell_t)[1].reshape(-1,1)], 1)*init_weight
    coord = torch.from_numpy(coord.astype(np.float32)).clone().requires_grad_()
    atom_num = len(coord)
    n_co = atom_num//2
    n_ni = atom_num - n_co

    ''' 
    Embed the corresponding params, also determines the atom specie
    '''
    p_nico = torch.tensor([
        [2.488746, 2.007018, 27.562015, 27.930410, 8.383453, 4.471175,
        0.429046, 0.633531, 0.443599, 0.820658, -2.693513, -0.076445, 0.241442,
        -2.375626, -2.7, 0, 0.265390, -0.152856, 0.469, -2.699486, 
        27.562015*0.85, 27.562015*1.15],
        [2.505979, 1.975299, 27.206789, 27.206789, 8.679625, 4.629134,
        0.421378, 0.640107, 0.5, 1, -2.541799, -0.219415, 0.733381, -1.589003,
        -2.56, 0, 0.705845, -0.687140, 0.694608, -2.559307,
        27.206789*0.85, 27.206789*1.15]
    ])
    ele_co = torch.cat((torch.zeros((n_co, 1)), torch.ones((n_co, 1))), dim=1)
    ele_ni = torch.cat((torch.ones((n_ni, 1)), torch.zeros((n_ni, 1))), dim=1)
    ele_list = torch.cat((ele_co, ele_ni), dim=0)
    shuffle_i = torch.randperm(ele_list.size(0))
    ele_list = ele_list[shuffle_i]

    #* Execution
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    param_ = torch.matmul(ele_list, p_nico).to(device)
    coord.to(device)
    date = '20221024'
    pth = f'/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/pyMD/2D_try/EAM/runs/{date}'
    m = model(coord, param_).to(device)
    # Instantiate optimizer
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma = 0.98)
    
    losses = train(m, opt, sch, pth, device, n=10)
    weight_ = m.weights.detach().cpu().numpy()
    np.save(pth, weight_)