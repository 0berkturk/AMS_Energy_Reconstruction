import torch
import torch.nn as nn
import numpy as np

class leak_double_matrix_exp_parameter(nn.Module): ## less than 3.6
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))
        self.exp_coeff1 = nn.Parameter(torch.zeros(18,72))
        self.exp_coeff2 = nn.Parameter(torch.zeros(18,72))

        self.p0=nn.Parameter(torch.ones(1))
        self.p1 = nn.Parameter(torch.ones(1))

    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1/((images/50000)+1e-2))
        exponential_term2 = torch.exp(self.exp_coeff2*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term1*exponential_term2

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3

        E_total_corrected = torch.sum(E_corrected_layers, 1)

        division = (self.p0 + self.p1*(E_corrected_layers[:, 16] + E_corrected_layers[:, 17])/E_total_corrected)

        E_total_corrected=E_total_corrected/division

        return E_total_corrected

class leak_dist_double_matrix_exp_parameter(nn.Module):  ## best one
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))
        self.exp_coeff1 = nn.Parameter(torch.zeros(18,72))
        self.exp_coeff2 = nn.Parameter(torch.zeros(18,72))

        self.p0=nn.Parameter(torch.ones(1))

        self.p1 = nn.Parameter(torch.ones(1))
        self.p2 = nn.Parameter(torch.ones(1))

        self.p3 = nn.Parameter(torch.ones(1))
        self.p4 = nn.Parameter(torch.ones(1))


        self.l0 = nn.Parameter(torch.ones(1))
        self.l1=nn.Parameter(torch.zeros(1))
        self.l2 = nn.Parameter(torch.zeros(1))
        self.l3 = nn.Parameter(torch.zeros(1))
        self.l4 = nn.Parameter(torch.zeros(1))

        self.k0 = nn.Parameter(torch.ones(1))
        self.k1 = nn.Parameter(torch.zeros(1))
        self.k2 = nn.Parameter(torch.zeros(1))
        self.k3 = nn.Parameter(torch.zeros(1))
        self.k4 = nn.Parameter(torch.zeros(1))

        self.c = nn.Parameter(torch.zeros(1))

    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1/((images/50000)+1e-2))
        exponential_term2 = torch.exp(self.exp_coeff2*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term1*exponential_term2

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3

        E_total_corrected = torch.sum(E_corrected_layers, 1)

        function = self.p1 * (torch.exp(self.p3 * E_total_corrected / 1000)) + self.p2 * (torch.exp(-self.p4 / ((E_total_corrected + 1e-2) / 1000)))

        division = (self.p0 + function * (E_corrected_layers[:, 16] + E_corrected_layers[:, 17]) / E_total_corrected)

        E_total_corrected = E_total_corrected / division


        E_total_corrected1=E_total_corrected/1000
        E_lost = self.c + self.l0*E_total_corrected*(self.l1*torch.exp(E_total_corrected1*self.l2)+self.l3*torch.exp(self.l4/E_total_corrected1)) + self.k0*E_total_corrected*(self.k1*torch.exp(E_total_corrected1*self.k2)+self.k3*torch.exp(self.k4/E_total_corrected1))
        E_total_corrected = E_total_corrected + E_lost

        return E_total_corrected

class leak_dist_matrix_single_exp_parameter2(nn.Module):  ## less than 3.6
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))
        self.exp_coeff1 = nn.Parameter(torch.zeros(18,72))

        self.p0=nn.Parameter(torch.ones(1))

        self.p1 = nn.Parameter(torch.ones(1))
        self.p2 = nn.Parameter(torch.ones(1))

        self.p3 = nn.Parameter(torch.ones(1))
        self.p4 = nn.Parameter(torch.ones(1))
    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term1

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3

        E_total_corrected = torch.sum(E_corrected_layers, 1)

        function = self.p1 * (torch.exp(self.p3 * E_total_corrected / 1000)) + self.p2 * (torch.exp(-self.p4 / ((E_total_corrected + 1e-2) / 1000)))

        division = (self.p0 + function * (E_corrected_layers[:, 16] + E_corrected_layers[:, 17]) / E_total_corrected)

        E_total_corrected = E_total_corrected / division

        return E_total_corrected


class leak_dist_matrix_single_exp_parameter1(nn.Module):  ## ıt was 4.5. now less than 3.9
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))
        self.exp_coeff1 = nn.Parameter(torch.zeros(18,72))

        self.p0=nn.Parameter(torch.ones(1))

        self.p1 = nn.Parameter(torch.ones(1))
        self.p2 = nn.Parameter(torch.ones(1))

        self.p3 = nn.Parameter(torch.ones(1))
        self.p4 = nn.Parameter(torch.ones(1))

    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1/((images/50000)+1e-1))
        E_cell_corrected = images*self.coefficient*exponential_term1

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3

        E_total_corrected = torch.sum(E_corrected_layers, 1)

        function = self.p1 * (torch.exp(self.p3 * E_total_corrected / 1000)) + self.p2 * (torch.exp(-self.p4 / ((E_total_corrected + 1e-2) / 1000)))

        division = (self.p0 + function * (E_corrected_layers[:, 16] + E_corrected_layers[:, 17]) / E_total_corrected)

        E_total_corrected = E_total_corrected / division
        return E_total_corrected


class leak_distribution_double_exp_parameter(nn.Module):  ## less than 7
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))
        self.exp_coeff1 = nn.Parameter(torch.zeros(1))
        self.exp_coeff2 = nn.Parameter(torch.zeros(1))

        self.p0=nn.Parameter(torch.ones(1))

        self.p1 = nn.Parameter(torch.ones(1))
        self.p2 = nn.Parameter(torch.ones(1))

        self.p3 = nn.Parameter(torch.ones(1))
        self.p4 = nn.Parameter(torch.ones(1))

    def forward(self,images):
        exponential_term1 = torch.exp(self.exp_coeff1/((images/50000)+1e-2))
        exponential_term2 = torch.exp(self.exp_coeff2*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term1*exponential_term2

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        E_total_corrected = torch.sum(E_corrected_layers, 1)

        function = self.p1 * (torch.exp(self.p3 * E_total_corrected / 1000)) + self.p2 * (torch.exp(-self.p4 / ((E_total_corrected + 1e-2) / 1000)))

        division = (self.p0 + function * (E_corrected_layers[:, 16] + E_corrected_layers[:, 17]) / E_total_corrected)

        E_total_corrected = E_total_corrected / division
        return E_total_corrected

class leak_distribution_single_exp_parameter2(nn.Module):  ## it is bad less than 7.
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))
        self.exp_coeff = nn.Parameter(torch.zeros(1))

        self.p0=nn.Parameter(torch.ones(1))

        self.p1 = nn.Parameter(torch.ones(1))
        self.p2 = nn.Parameter(torch.ones(1))

        self.p3 = nn.Parameter(torch.ones(1))
        self.p4 = nn.Parameter(torch.ones(1))

    def forward(self,images):
        exponential_term = torch.exp(self.exp_coeff*(images/50000))
        E_cell_corrected = images*self.coefficient*exponential_term

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        E_total_corrected = torch.sum(E_corrected_layers, 1)

        function = self.p1 * (torch.exp(self.p3 * E_total_corrected / 1000)) + self.p2 * (torch.exp(-self.p4 / ((E_total_corrected + 1e-2) / 1000)))

        division = (self.p0 + function * (E_corrected_layers[:, 16] + E_corrected_layers[:, 17]) / E_total_corrected)

        E_total_corrected = E_total_corrected / division

        return E_total_corrected

class leak_distribution_single_exp_parameter1(nn.Module):  ## it is bad. less than 17. now it is less than 7
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))
        self.exp_coeff = nn.Parameter(torch.zeros(1))

        self.p0=nn.Parameter(torch.ones(1))

        self.p1 = nn.Parameter(torch.ones(1))
        self.p2 = nn.Parameter(torch.ones(1))

        self.p3 = nn.Parameter(torch.ones(1))
        self.p4 = nn.Parameter(torch.ones(1))

    def forward(self,images):
        exponential_term = torch.exp(self.exp_coeff/((images/50000)+1e-2))
        E_cell_corrected = images*self.coefficient*exponential_term

        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        E_total_corrected = torch.sum(E_corrected_layers, 1)

        function = self.p1 * (torch.exp(self.p3 * E_total_corrected / 1000)) + self.p2 * (torch.exp(-self.p4 / ((E_total_corrected + 1e-2) / 1000)))

        division = (self.p0 + function * (E_corrected_layers[:, 16] + E_corrected_layers[:, 17]) / E_total_corrected)

        E_total_corrected = E_total_corrected / division

        return E_total_corrected


class leak_distribuiton_matrix_parameter(nn.Module):  ## normally, it is gives 6.25.  with leakage, less than 4.3
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))

        self.p0=nn.Parameter(torch.ones(1))

        self.p1 = nn.Parameter(torch.ones(1))
        self.p2 = nn.Parameter(torch.ones(1))

        self.p3 = nn.Parameter(torch.ones(1))
        self.p4 = nn.Parameter(torch.ones(1))

    def forward(self,images):
        E_cell_corrected = images*self.coefficient
        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        E_total_corrected = torch.sum(E_corrected_layers, 1)

        function = self.p1 * (torch.exp(self.p3 * E_total_corrected / 1000)) + self.p2 * (torch.exp(-self.p4 / ((E_total_corrected + 1e-2) / 1000)))

        division = (self.p0 + function * (E_corrected_layers[:, 16] + E_corrected_layers[:, 17]) / E_total_corrected)

        E_total_corrected = E_total_corrected / division

        return E_total_corrected

class leak_matrix_parameter(nn.Module):  ## normally, it is gives 6.25.  with leakage, less than 5.4
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(18,72))

        self.p0=nn.Parameter(torch.ones(1))
        self.p1 = nn.Parameter(torch.ones(1))

    def forward(self,images):
        E_cell_corrected = images*self.coefficient
        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        E_total_corrected = torch.sum(E_corrected_layers, 1)

        division = (self.p0 + self.p1*(E_corrected_layers[:, 16] + E_corrected_layers[:, 17])/E_total_corrected)

        E_total_corrected = E_total_corrected / division

        return E_total_corrected


class leak_distribution_single_parameter(nn.Module):  # less than 7.5
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))

        self.p0=nn.Parameter(torch.ones(1))

        self.p1 = nn.Parameter(torch.ones(1))
        self.p2 = nn.Parameter(torch.ones(1))

        self.p3 = nn.Parameter(torch.ones(1))
        self.p4 = nn.Parameter(torch.ones(1))

    def forward(self,images):
        E_cell_corrected = images*self.coefficient
        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        E_total_corrected = torch.sum(E_corrected_layers, 1)

        function = self.p1*(torch.exp(self.p3*E_total_corrected/1000)) + self.p2*(torch.exp(-self.p4/((E_total_corrected+1e-2)/1000)))
        division = (self.p0 + function*(E_corrected_layers[:, 16] + E_corrected_layers[:, 17])/E_total_corrected)

        E_total_corrected=E_total_corrected/division



        return E_total_corrected


class leak_single_parameter(nn.Module):  # less than 10
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))

        self.p0=nn.Parameter(torch.ones(1))
        self.p1 = nn.Parameter(torch.ones(1))

        self.l0 = nn.Parameter(torch.ones(1))
        self.l1=nn.Parameter(torch.zeros(1))
        self.l2 = nn.Parameter(torch.zeros(1))
        self.l3 = nn.Parameter(torch.zeros(1))
        self.l4 = nn.Parameter(torch.zeros(1))

        self.k0 = nn.Parameter(torch.ones(1))
        self.k1 = nn.Parameter(torch.zeros(1))
        self.k2 = nn.Parameter(torch.zeros(1))
        self.k3 = nn.Parameter(torch.zeros(1))
        self.k4 = nn.Parameter(torch.zeros(1))

        self.c = nn.Parameter(torch.zeros(1))


    def forward(self,images):
        E_cell_corrected = images*self.coefficient
        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3
        E_total_corrected = torch.sum(E_corrected_layers, 1)

        division = (self.p0 + self.p1*(E_corrected_layers[:, 16] + E_corrected_layers[:, 17])/E_total_corrected)

        E_total_corrected=E_total_corrected/division

        E_total_corrected1=E_total_corrected/1000
        E_lost = self.c + self.l0*E_total_corrected*(self.l1*torch.exp(E_total_corrected1*self.l2)+self.l3*torch.exp(self.l4/E_total_corrected1)) + self.k0*E_total_corrected*(self.k1*torch.exp(E_total_corrected1*self.k2)+self.k3*torch.exp(self.k4/E_total_corrected1))
        E_total_corrected = E_total_corrected + E_lost
        return E_total_corrected