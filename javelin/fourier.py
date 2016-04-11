"""This module define the Structure object"""
import numpy as np


class Fourier(object):
    __radiation_type = ["neutrons", "xray", "electrons"]

    def __init__(self):
        self.structure = None
        self.radiation = 'neutrons'
        self.wavelenght = 1.54
        self.lots = None
        self.average = 0.0
        self.na = 101
        self.no = 101
        self.ll = np.array([0.0, 0.0, 0.0])
        self.lr = np.array([2.0, 0.0, 0.0])
        self.ul = np.array([0.0, 2.0, 0.0])

    def calculate(self):
        """Returns a Data object"""
        output_array = np.zeros([self.na, self.no], dtype=np.complex)
        vector1_step = (self.lr - self.ll)/(self.na-1)
        vector2_step = (self.ul - self.ll)/(self.no-1)
        kx = np.zeros([self.na, self.no])
        ky = np.zeros([self.na, self.no])
        kz = np.zeros([self.na, self.no])
        for x in range(self.na):
            for y in range(self.no):
                v = self.ll + x*vector1_step + y*vector2_step
                kx[x, y] = v[0]
                ky[x, y] = v[1]
                kz[x, y] = v[2]
        kx *= (2*np.pi)
        ky *= (2*np.pi)
        kz *= (2*np.pi)
        for atom in self.structure.get_scaled_positions():
            dot = kx*atom[0] + ky*atom[1] + kz*atom[2]
            output_array += np.exp(dot*1j)
        results = np.real(output_array*np.conj(output_array))
        return results

    def calculate_fast(self):
        """Returns a Data object"""
        output_array = np.zeros([self.na, self.no], dtype=np.complex)
        kx, ky, kz = calc_k_grid(self.ll, self.lr, self.ul, self.na, self.no)
        kx *= (2*np.pi)
        ky *= (2*np.pi)
        kz *= (2*np.pi)
        for atom in self.structure.get_scaled_positions():
            dotx = np.exp(kx*atom[0]*1j)
            doty = np.exp(ky*atom[1]*1j)
            dotz = np.exp(kz*atom[2]*1j)
            sumexp = dotx * doty * dotz
            output_array += sumexp
        results = np.real(output_array*np.conj(output_array))
        return results


def calc_k_grid(ll, lr, ul, na, no):
    va = lr - ll
    vo = ul - ll
    vector1_step = (lr - ll)/(na-1)
    vector2_step = (ul - ll)/(no-1)
    kx_bina, kx_bino = get_bin_number(va, vo, na, no, 0)
    ky_bina, ky_bino = get_bin_number(va, vo, na, no, 1)
    kz_bina, kz_bino = get_bin_number(va, vo, na, no, 2)
    kx = np.zeros([kx_bina, kx_bino])
    ky = np.zeros([ky_bina, ky_bino])
    kz = np.zeros([kz_bina, kz_bino])
    for x in range(kx_bina):
        for y in range(kx_bino):
            v = ll + x*vector1_step + y*vector2_step
            kx[x, y] = v[0]
    for x in range(ky_bina):
        for y in range(ky_bino):
            v = ll + x*vector1_step + y*vector2_step
            ky[x, y] = v[1]
    for x in range(kz_bina):
        for y in range(kz_bino):
            v = ll + x*vector1_step + y*vector2_step
            kz[x, y] = v[2]
    return kx, ky, kz


def get_bin_number(va, vo, na, no, index):
    if va[index] == 0:
        binx = 1
    else:
        binx = na
    if vo[index] == 0:
        biny = 1
    else:
        biny = no
    return binx, biny
