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


class attn_single_parameter(nn.Module): # negllect opposite attn
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))
        self.same_attn_coefficient = nn.Parameter(torch.zeros(1))
        self.opposite_attn_coefficient = nn.Parameter(torch.ones(1))

        self.same_attn_exp_coefficient1 = nn.Parameter(torch.ones(1))
        self.same_attn_exp_coefficient2 = nn.Parameter(torch.ones(1))

        self.opposite_attn_exp_coefficient1 = nn.Parameter(torch.ones(1))
        self.opposite_attn_exp_coefficient2 = nn.Parameter(torch.ones(1))

    def forward(self,images,orders):
        opposite_dir = torch.zeros(images.shape[0],18,72).to("cuda")
        for i in range(0,16):
            opposite_dir[:,i,:] = orders[:,i+2,:]

        opposite_dir[:, 16, :] = orders[:, 15, :]
        opposite_dir[:, 17, :] = orders[:, 15, :]
        opposite_dir=opposite_dir/17
        E_cell_corrected = images*self.coefficient

        # first make atten matrix
        same_dir_attn_matrix = self.same_attn_coefficient*torch.exp(-orders/self.same_attn_exp_coefficient1) + (1-self.same_attn_coefficient)*torch.exp(-orders/self.same_attn_exp_coefficient2)
        opposite_dir_attn_matrix = self.opposite_attn_coefficient * torch.exp(-opposite_dir / self.opposite_attn_exp_coefficient1) + (1 - self.opposite_attn_coefficient) * torch.exp(-opposite_dir / self.opposite_attn_exp_coefficient2)
        #for i in range(0,16):
            #E_cell_corrected[:,i,:] = E_cell_corrected[:,i,:]*opposite_dir_attn_matrix[:,i+2,:]

        #E_cell_corrected[:, 16, :] = E_cell_corrected[:, 16 , :] * opposite_dir_attn_matrix[:, 15, :]
        #E_cell_corrected[:, 17, :] = E_cell_corrected[:, 17, :] * opposite_dir_attn_matrix[:, 15, :]

        E_cell_corrected =E_cell_corrected*opposite_dir_attn_matrix
        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3

        E_total_corrected = torch.sum(E_corrected_layers, 1)

        return E_total_corrected

class attn_single_parameter2(nn.Module): # it is wrong, long
    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.ones(1))
        self.same_attn_coefficient = nn.Parameter(torch.ones(1))
        self.opposite_attn_coefficient = nn.Parameter(torch.ones(1))

        self.same_attn_exp_coefficient1 = nn.Parameter(torch.ones(1))
        self.same_attn_exp_coefficient2 = nn.Parameter(torch.ones(1))

        self.opposite_attn_exp_coefficient1 = nn.Parameter(torch.ones(1))
        self.opposite_attn_exp_coefficient2 = nn.Parameter(torch.ones(1))

    def forward(self,images,orders):
        E_cell_corrected = images*self.coefficient
        order_x1= (orders[:,0,:] + orders[:,1,:])/2
        order_y1 = (orders[:,2 , :] + orders[:, 3, :]) / 2

        order_x2 = (orders[:, 4, :] + orders[:, 5, :]) / 2
        order_y2 = (orders[:, 6, :] + orders[:, 7, :]) / 2

        order_x3= (orders[:, 8, :] + orders[:, 9, :]) / 2
        order_y3 = (orders[:, 10, :] + orders[:, 11, :]) / 2

        order_x4 = (orders[:, 12, :] + orders[:, 13, :]) / 2
        order_y4 = (orders[:, 14, :] + orders[:, 15, :]) / 2

        order_x5 = (orders[:, 16, :] + orders[:, 17, :]) / 2

        #E_cell_corrected[:,0,:] = E_cell_corrected[:,0,:]*((self.same_attn_coefficient)*() + (1-self.same_attn_coefficient)*() ) * ((self.opposite_attn_coefficient)*() + (1-self.opposite_attn_coefficient)*() )

        i=0
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :]* ((self.same_attn_coefficient)*(torch.exp(order_x1/self.same_attn_exp_coefficient1))+ (1-self.same_attn_coefficient)*(torch.exp(order_x1/self.same_attn_exp_coefficient2)))\
                                    *((self.opposite_attn_coefficient) * (torch.exp(order_y1/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y1/self.opposite_attn_exp_coefficient2)))
        E_cell_corrected[:, i+1, :] = E_cell_corrected[:, i+1, :] * ((self.same_attn_coefficient) * (torch.exp(order_x1 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x1 / self.same_attn_exp_coefficient2)))\
                                                                  *((self.opposite_attn_coefficient) * (torch.exp(order_y1/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y1/self.opposite_attn_exp_coefficient2)))

        i = 2
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :] * ((self.same_attn_coefficient) * (torch.exp(order_y1 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_y1 / self.same_attn_exp_coefficient2)))\
                                                              *((self.opposite_attn_coefficient) * torch.exp(order_x2/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_x2/self.opposite_attn_exp_coefficient2))
        E_cell_corrected[:, i + 1, :] = E_cell_corrected[:, i + 1, :] * ((self.same_attn_coefficient) * (torch.exp(order_y1/ self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_y1 / self.same_attn_exp_coefficient2)))\
                                                               *((self.opposite_attn_coefficient) * torch.exp(order_x2/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_x2/self.opposite_attn_exp_coefficient2))


        i = 4
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :] * ((self.same_attn_coefficient) * (torch.exp(order_x2 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x2 / self.same_attn_exp_coefficient2)))\
                                                               *((self.opposite_attn_coefficient) * torch.exp(order_y2/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y2/self.opposite_attn_exp_coefficient2))
        E_cell_corrected[:, i + 1, :] = E_cell_corrected[:, i + 1, :] * ((self.same_attn_coefficient) * (torch.exp(order_x2 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x2 / self.same_attn_exp_coefficient2)))\
                                                                    *((self.opposite_attn_coefficient) * torch.exp(order_y2/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y2/self.opposite_attn_exp_coefficient2))


        i = 6
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :] * ((self.same_attn_coefficient) * (torch.exp(order_y2 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_y2 / self.same_attn_exp_coefficient2)))\
                                    *((self.opposite_attn_coefficient) * torch.exp(order_x3/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_x3/self.opposite_attn_exp_coefficient2))
        E_cell_corrected[:, i + 1, :] = E_cell_corrected[:, i + 1, :] * ((self.same_attn_coefficient) * (torch.exp(order_y2 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_y2 / self.same_attn_exp_coefficient2)))\
                                        *((self.opposite_attn_coefficient) * torch.exp(order_x3/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_x3/self.opposite_attn_exp_coefficient2))

        i = 8
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :] * ((self.same_attn_coefficient) * (torch.exp(order_x3 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x3 / self.same_attn_exp_coefficient2)))\
                                    *((self.opposite_attn_coefficient) * torch.exp(order_y3/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y3/self.opposite_attn_exp_coefficient2))
        E_cell_corrected[:, i + 1, :] = E_cell_corrected[:, i + 1, :] * ((self.same_attn_coefficient) * (torch.exp(order_x3 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x3 / self.same_attn_exp_coefficient2)))\
                                        *((self.opposite_attn_coefficient) * torch.exp(order_y3/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y3/self.opposite_attn_exp_coefficient2))

        i = 10
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :] * ((self.same_attn_coefficient) * (torch.exp(order_y3 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_y3  / self.same_attn_exp_coefficient2)))\
                                    *((self.opposite_attn_coefficient) * torch.exp(order_x4/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_x4/self.opposite_attn_exp_coefficient2))
        E_cell_corrected[:, i + 1, :] = E_cell_corrected[:, i + 1, :] * ((self.same_attn_coefficient) * (torch.exp(order_y3  / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_y3  / self.same_attn_exp_coefficient2)))\
                                        *((self.opposite_attn_coefficient) * torch.exp(order_x4/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_x4/self.opposite_attn_exp_coefficient2))


        i = 12
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :] * ((self.same_attn_coefficient) * (torch.exp(order_x4  / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x4  / self.same_attn_exp_coefficient2)))\
                                    *((self.opposite_attn_coefficient) * torch.exp(order_y4/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y4/self.opposite_attn_exp_coefficient2))
        E_cell_corrected[:, i + 1, :] = E_cell_corrected[:, i + 1, :] * ((self.same_attn_coefficient) * (torch.exp(order_x4  / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x4  / self.same_attn_exp_coefficient2)))\
                                        *((self.opposite_attn_coefficient) * torch.exp(order_y4/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y4/self.opposite_attn_exp_coefficient2))

        i = 14
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :] * ((self.same_attn_coefficient) * (torch.exp(order_y4 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_y4 / self.same_attn_exp_coefficient2)))\
                                    *((self.opposite_attn_coefficient) * torch.exp(order_x5/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_x5/self.opposite_attn_exp_coefficient2))
        E_cell_corrected[:, i + 1, :] = E_cell_corrected[:, i + 1, :] * ((self.same_attn_coefficient) * (torch.exp(order_y4 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_y4 / self.same_attn_exp_coefficient2)))\
                                        *((self.opposite_attn_coefficient) * torch.exp(order_x5/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_x5/self.opposite_attn_exp_coefficient2))

        i = 16
        E_cell_corrected[:, i, :] = E_cell_corrected[:, i, :] * ((self.same_attn_coefficient) * (torch.exp(order_x5 / self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x5/ self.same_attn_exp_coefficient2)))*((self.opposite_attn_coefficient) * torch.exp(order_y4/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y4/self.opposite_attn_exp_coefficient2))
        E_cell_corrected[:, i + 1, :] = E_cell_corrected[:, i + 1, :] * ((self.same_attn_coefficient) * (torch.exp(order_x5/ self.same_attn_exp_coefficient1)) + (1 - self.same_attn_coefficient) * (torch.exp(order_x5 / self.same_attn_exp_coefficient2)))*((self.opposite_attn_coefficient) * torch.exp(order_y4/self.opposite_attn_exp_coefficient1))+ (1-self.opposite_attn_coefficient)*(torch.exp(order_y4/self.opposite_attn_exp_coefficient2))


        E_corrected_layers = torch.sum(E_cell_corrected, 2) * 1e-3

        E_total_corrected = torch.sum(E_corrected_layers, 1)
        print(E_cell_corrected)

        return E_total_corrected