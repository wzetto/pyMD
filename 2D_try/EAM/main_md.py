import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch import nn
from torch.functional import F
from copy import copy
from itertools import combinations
import time
from torch.profiler import profile, record_function, ProfilerActivity
from scipy import constants
import os
import torch.multiprocessing as tmp
from torch.utils.tensorboard import SummaryWriter

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created Directory : ", directory)
    else:
        print("Directory already existed : ", directory)
    return directory

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
    def __init__(self, atom_list, param_list, mass_list, v_list, device):
        super().__init__()
        '''
        atom_list: N*dimension
        param_list: N*number of params per each atom
        per each row:
        r_e, f_e, rho_e, rho_s, alpha, beta, A, B, kappa, lamda,
        f_n0, f_n1, f_n2, f_n3, f_0, f_1, f_2, f_3, eta, f_e
        '''
        #* row1: x-coord, row2: y-coord
        self.weights = atom_list
        self.weights_ = self.weights.clone()
        atom_num = len(atom_list)

        self.ind_inter = torch.combinations(torch.arange(self.weights.size(0)), r=2).to(device)
        ind_inter_f = torch.fliplr(self.ind_inter)
        ind_all = torch.cat((self.ind_inter, ind_inter_f), dim=0)
        ind_all = ind_all[ind_all[:, 0].sort()[1]]
        self.ind_block = torch.split(ind_all, atom_num-1)

        self.rho_list = torch.zeros(atom_num).to(device)
        self.pe_list = torch.zeros(atom_num).to(device)
        self.acc_list = torch.zeros(atom_num, self.weights.size(1)).to(device)
        self.v_list = v_list
        self.mass = mass_list

        self.range = torch.arange(atom_num).to(device)
        self.params = param_list
        self.opt = torch.optim.Adam([self.weights])

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

    def single_block(self, r_ind):
        '''
        Extract each atom block containing the centre i and surronding j.
        r_ind corresponds to the index of centre atom.
        '''
        block = self.ind_block[r_ind]
        coord_block = self.weights[block]
        param_block = self.params[block]

        delta_xy = coord_block[:,1]-coord_block[:,0]
        r_blo = torch.norm(delta_xy, dim=1)
        #* So this will be the cutoff ~3 NN?
        effe_r_ind = torch.nonzero(r_blo <= 8.5)
        r_blo = r_blo[torch.flatten(effe_r_ind)]
        r_blo.requires_grad_()
        param_block = param_block[torch.flatten(effe_r_ind)]
        delta_xy = delta_xy[torch.flatten(effe_r_ind)]

        #* Potential energy
        
        fr_0 = self.f_r(r_blo, param_block[:, 0][:, [1, 5, 0, 9]]) #* For f^0(r)
        fr_1 = self.f_r(r_blo, param_block[:, 1][:, [1, 5, 0, 9]]) #* For f^1(r)
        phir_0 = self.phi_r(r_blo, param_block[:, 0][:, [6,7,0,4,5,8,9]]) #* For phi^0(r)
        phir_1 = self.phi_r(r_blo, param_block[:, 1][:, [6,7,0,4,5,8,9]]) #* For phi^1(r)
        phi_01 = 1/2*(fr_1/fr_0*phir_0 + fr_0/fr_1*phir_1) #* For phi^01
        p_e_ = torch.sum(phi_01) #* Potential energy term

        #* Electronic density
        rho_i = torch.sum(self.f_r(r_blo, param_block[:, 1][:, [1, 5, 0, 9]])) #* For f^1(r)
        f_rho_ = self.f_rho(rho_i, param_block[:, 0][:, [10,11,12,13,14,15,16,17,19,20,2,21,3,18]][0]) #* For F(rho)

        # self.rho_list[r_ind] = f_rho_
        # self.pe_list[r_ind] = p_e_
        e_x = f_rho_ + 1/2*p_e_

        e_x.backward(retain_graph = True)
        acc = torch.sum(r_blo.grad.reshape(-1,1)*delta_xy/r_blo.reshape(-1,1), dim=0)/self.mass[r_ind] #*N*2
        # self.acc_list[r_ind] = acc
        r_blo.detach()

        return f_rho_.detach(), p_e_.detach(), acc.detach(), r_ind

    def grad_calc(self, process_num):
        ''' 
        Calculate the whole energy with keeping gradient
        '''

        # with concurrent.futures.ProcessPoolExecutor(max_workers=process_num) as executor:
        #     outputs = executor.map(self.single_block, self.range)
        pool = tmp.Pool(processes = process_num)
        outputs = pool.map(self.single_block, self.range)

        for output in outputs:
            ind = output[-1]
            self.rho_list[ind] = output[0]
            self.pe_list[ind] = output[1]
            self.acc_list[ind] = output[2]

        # print(p_e, self.rho_list)
        self.e_total = torch.sum(self.rho_list) + 1/2*torch.sum(self.pe_list)


def train(model_, path_save, dt, temp_given, alpha, lo_b, up_b, n_core=12, device=None, n = 1000):

    writer = SummaryWriter(log_dir = path_save)
    create_dir(path_save+'/.gitignore')
    length = len(model_.weights)
    k_b = constants.k
    ev_j = constants.physical_constants['atomic unit of charge'][0]
    ''' 
    v_list: angstrom / s
    acc: angstrom / s^2
    e_total: eV
    kinetic, potential energy in Tensorboard: eV
    '''

    for i in range(n):
        #* Step 1
        model_.grad_calc(n_core)
        model_.acc_list *= (ev_j*1e20) #* eV -> J
        with torch.no_grad():
            model_.weights += (model_.v_list*dt + 1/2*model_.acc_list*dt**2) #* x-step
            model_.weights = clamp(lo_b, up_b, model_.weights)
        model_.v_list += 1/2*model_.acc_list*dt
        model_.v_list -= torch.sum(model_.v_list, 0)/length

        #* Step 2
        model_.grad_calc(n_core)
        model_.acc_list *= (ev_j*1e20)
        model_.v_list += 1/2*model_.acc_list*dt
        model_.v_list -= torch.sum(model_.v_list, 0)/length
        k_e_ = 1/2*torch.sum(model_.mass.reshape(-1,1)*(model_.v_list*1e-10)**2)
        temp_ = k_e_/(3/2*(length-1))/k_b

        #* Load status of config.
        clear_output(True)
        weight_ = m.weights.detach().cpu().numpy()
        fig, ax = plt.subplots()
        plt.scatter(weight_[:, 0], weight_[:, 1])
        plt.xlim(-2, 23.5)
        plt.ylim(-2, 14.5)
        plt.text(17, 13, f'{i} fs', fontsize=20)
        fig.savefig(path_save+f'/.gitignore/{i}.png',)
        f = plt.figure()
        f.clear()
        plt.close(f)

        writer.add_scalar("Potential energy", model_.e_total, i)
        writer.add_scalar("Kinetic energy", k_e_/ev_j, i)
        writer.add_scalar("Temperature", temp_, i)

        if i%100 == 0:
            s_adjust = torch.sqrt((temp_given+(temp_-temp_given)*alpha)/temp_)
            model_.v_list *= s_adjust

            print(i)

            # clear_output(True)

#! Check latent bug (?)

if __name__ == '__main__':
    use_relax = True
    ''' 
    Generate the primary CoNi cell in planar FCC lattice
    '''
    x_extend, y_extend = 5, 5
    r_equ = 2.363925
    cell = supercell_gen(x_extend, y_extend)
    cell_t = np.array(cell, dtype=bool)
    init_weight = np.array([math.sqrt(3)/2*r_equ, 1/2*r_equ])
    coord = np.concatenate([np.where(cell_t)[0].reshape(-1,1),
                            np.where(cell_t)[1].reshape(-1,1)], 1)*init_weight

    if use_relax:
        coord = np.load('/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/pyMD_buffer/EAM_runs/20221026_relax/weight_5_5.npy')
        coord -= coord[0] #* Normalization

    coord = torch.from_numpy(coord.astype(np.float32)).clone()
    atom_num = len(coord)
    atom_dim = coord.size(1)

    n_co = atom_num//2
    n_ni = atom_num - n_co

    ''' 
    Set upper and lower bound for matrix
    '''
    x_upper, x_lower = (x_extend+1/2)*r_equ*math.sqrt(3), -(1/2)*r_equ*math.sqrt(3)
    y_upper, y_lower = (y_extend+1/2)*r_equ, -(1/2)*r_equ

    x_u = torch.ones(atom_num)*x_upper
    x_l = torch.ones(atom_num)*x_lower

    y_u = torch.ones(atom_num)*y_upper
    y_l = torch.ones(atom_num)*y_lower

    xy_l = torch.cat((x_l.reshape(-1,1), y_l.reshape(-1,1)), dim=1).long()
    xy_u = torch.cat((x_u.reshape(-1,1), y_u.reshape(-1,1)), dim=1).long()

    def clamp(lo_b, up_b, a):
        a = torch.where(a > lo_b, a, a+up_b-lo_b)
        a = torch.where(a < up_b, a, a-up_b+lo_b)
        return a

    ''' 
    Embed the corresponding params, also determines the atom specie
    '''
    p_nico = torch.tensor([
        [2.488746, 2.007018, 27.562015, 27.930410, 8.383453, 4.471175,
        0.429046, 0.633531, 0.443599, 0.820658, -2.693513, -0.076445, 0.241442,
        -2.375626, -2.7, 0, 0.265390, -0.152856, 0.469, -2.699486, 
        27.562015*0.85, 27.562015*1.15], #* Ni
        [2.505979, 1.975299, 27.206789, 27.206789, 8.679625, 4.629134,
        0.421378, 0.640107, 0.5, 1, -2.541799, -0.219415, 0.733381, -1.589003,
        -2.56, 0, 0.705845, -0.687140, 0.694608, -2.559307,
        27.206789*0.85, 27.206789*1.15] #* Co
    ])
    m_nico = torch.tensor([ 
        [58.6934],
        [58.9332]
    ])*constants.u

    ele_co = torch.cat((torch.zeros((n_co, 1)), torch.ones((n_co, 1))), dim=1)
    ele_ni = torch.cat((torch.ones((n_ni, 1)), torch.zeros((n_ni, 1))), dim=1)
    ele_list = torch.cat((ele_co, ele_ni), dim=0)
    shuffle_i = torch.randperm(ele_list.size(0))
    ele_list = ele_list[shuffle_i]
    if use_relax:
        ele_list = np.load('/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/pyMD_buffer/EAM_runs/20221026_relax/ele_list_5_5.npy')
        ele_list = torch.from_numpy(ele_list.astype(np.float32))

    #* Execution
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    param_ = torch.matmul(ele_list, p_nico)
    mass_ = torch.flatten(torch.matmul(ele_list, m_nico))

    #* Velocity
    temp = 77
    k_b = constants.k
    v_list = (torch.rand(atom_num, atom_dim)*4-2)*torch.sqrt(3*(1-1/atom_dim)*k_b*temp/mass_.reshape(-1,1))*1e10
    v_list -= torch.sum(v_list, 0)/atom_num

    #* To device
    v_list = v_list.to(device)
    mass_ = mass_.to(device)
    param_ = param_.to(device)
    coord = coord.to(device)

    localtime = time.localtime(time.time())
    yr_, m_, d_ = localtime[:3]
    date = f'{yr_}{m_}{d_}_MD'
    pth = f'/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/pyMD_buffer/EAM_runs/{date}'

    m = model(coord, param_, mass_, v_list, device).to(device)

    dt = torch.tensor(1e-15).to(device) #* Time step, 1 fs = 1e-15 s
    alpha = 0.75 #* For temperature adjusting

    #* Main
    train(m, pth, dt, temp, alpha, xy_l, xy_u, n_core=22, device=device, n=10000)

    #*Store
    weight_ = m.weights.detach().cpu().numpy()
    weight_raw = m.weights_.detach().cpu().numpy()
    # plt.scatter(weight_[:, 0], weight_[:, 1])
    np.save(pth+'/weight.npy', weight_)
    np.save(pth+'/weight_raw.npy', weight_raw)
    np.save(pth+'/ele_specie.npy', ele_list)