import torch
import torch.nn as nn
import numpy as np


class double_matrix_exp_parameter(nn.Module):  ## best one
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))
        self.exp_coeff1 = nn.Parameter(torch.zeros(18,72))
        self.exp_coeff2 = nn.Parameter(torch.zeros(18,72))

    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1/((images/50000)+1e-2))
        exponential_term2 = torch.exp(self.exp_coeff2*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term1*exponential_term2

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        #print(images)
        #print(E_cell_corrected)
        #print(self.coefficient,self.exp_coeff)
        E_total_corrected = torch.sum(E_corrected_layers, 1)
        return E_total_corrected

class matrix_single_exp_parameter2(nn.Module):  ##
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))
        self.exp_coeff1 = nn.Parameter(torch.zeros(18,72))


    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term1

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        #print(images)
        #print(E_cell_corrected)
        #print(self.coefficient,self.exp_coeff)
        E_total_corrected = torch.sum(E_corrected_layers, 1)

        return E_total_corrected


class matrix_single_exp_parameter1(nn.Module):  ##
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))
        self.exp_coeff1 = nn.Parameter(torch.zeros(18,72))


    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1/((images/50000)+1e-1))
        E_cell_corrected = images*self.coefficient*exponential_term1

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        #print(images)
        #print(E_cell_corrected)
        #print(self.coefficient,self.exp_coeff)
        E_total_corrected = torch.sum(E_corrected_layers, 1)
        return E_total_corrected


class double_exp_parameter(nn.Module):  ## it is bad.
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))
        self.exp_coeff1 = nn.Parameter(torch.zeros(1))
        self.exp_coeff2 = nn.Parameter(torch.zeros(1))

    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1/((images/50000)+1e-2))
        exponential_term2 = torch.exp(self.exp_coeff2*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term1*exponential_term2

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        #print(images)
        #print(E_cell_corrected)
        #print(self.coefficient,self.exp_coeff)
        E_total_corrected = torch.sum(E_corrected_layers, 1)
        return E_total_corrected

class single_exp_parameter2(nn.Module):  ## it is bad.
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))
        self.exp_coeff = nn.Parameter(torch.zeros(1))

    def forward(self,images):
        exponential_term = torch.exp(self.exp_coeff*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        #print(images)
        #print(E_cell_corrected)
        #print(self.coefficient,self.exp_coeff)
        E_total_corrected = torch.sum(E_corrected_layers, 1)
        return E_total_corrected

class single_exp_parameter1(nn.Module):  ## it is bad.
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))
        self.exp_coeff = nn.Parameter(torch.zeros(1))

    def forward(self,images):
        exponential_term = torch.exp(self.exp_coeff/((images/50000)+1e-2))
        E_cell_corrected = images*self.coefficient*exponential_term

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        #print(images)
        #print(E_cell_corrected)
        #print(self.coefficient,self.exp_coeff)
        E_total_corrected = torch.sum(E_corrected_layers, 1)
        return E_total_corrected


class matrix_parameter(nn.Module):  ## gave +- 83 GeV, or 70000
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))

    def forward(self,images):
        E_cell_corrected = images*self.coefficient
        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        #print(images)
        #print(E_cell_corrected)
        #print(self.coefficient)
        E_total_corrected = torch.sum(E_corrected_layers, 1) * 1e-3
        return E_total_corrected

class single_parameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))

    def forward(self,images):
        E_cell_corrected = images*self.coefficient
        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        #print(images)
        #print(E_cell_corrected)
        #print(self.coefficient)
        E_total_corrected = torch.sum(E_corrected_layers, 1)
        return E_total_corrected