"""
======
mccore
======
"""

from libc.math cimport exp
from .energies cimport Energy
from .modifier cimport BaseModifier
from .random cimport random
cimport cython

cdef class Target:
    """Class to hold an Energy object with it associated neighbours"""
    cdef readonly int number_of_neighbours
    cdef Py_ssize_t[:,:] neighbours
    cdef Energy energy
    def __init__(self, Py_ssize_t[:,:] neighbours, Energy energy):
        assert neighbours.shape[1] == 5
        self.energy = energy
        self.neighbours = neighbours
        self.number_of_neighbours = len(self.neighbours)
    def __str__(self):
        return "{}(number_of_neighbours={})".format(self.__class__.__name__,self.number_of_neighbours)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef int mcrun(BaseModifier modifier, Target[:] targets,
                    int iterations, double temperature,
                    long[:,:,:,::1] a, double[:,:,:,::1] x, double[:,:,:,::1] y, double[:,:,:,::1] z):
    """This function is not meant to be used directly. It is used by
    :obj:`javelin.mc.MC`. The function does very little validation
    of the input values, it you don't provide exactly what is expected
    then segmentation fault is likely."""
    assert tuple(a.shape) == tuple(x.shape)
    assert tuple(a.shape) == tuple(y.shape)
    assert tuple(a.shape) == tuple(z.shape)
    cdef Py_ssize_t mod_x, mod_y, mod_z
    cdef int number_of_targets, number_of_cells
    cdef int not_accepted = 0
    cdef Py_ssize_t cell_x_target, cell_y_target, cell_z_target, ncell
    cdef Py_ssize_t[:, :] cells
    cdef Py_ssize_t target_number, neighbour, number_of_neighbours
    cdef double e0, e1, de
    cdef Energy energy
    cdef Target target
    cdef Py_ssize_t[:,:] neighbours
    number_of_targets = targets.shape[0]
    number_of_cells = modifier.number_of_cells
    mod_x = a.shape[0]
    mod_y = a.shape[1]
    mod_z = a.shape[2]
    for _ in range(iterations):
        cells = modifier.get_random_cells(a.shape[0], a.shape[1], a.shape[2])
        e0 = 0
        for target_number in range(number_of_targets):
            target = targets[target_number]
            neighbours = target.neighbours
            energy = target.energy
            number_of_neighbours = target.number_of_neighbours
            for ncell in range(number_of_cells):
                for neighbour in range(number_of_neighbours):
                    cell_x_target = (cells[ncell,0]+neighbours[neighbour,2]) % mod_x
                    cell_y_target = (cells[ncell,1]+neighbours[neighbour,3]) % mod_y
                    cell_z_target = (cells[ncell,2]+neighbours[neighbour,4]) % mod_z
                    e0 += energy.evaluate(a[cells[ncell,0], cells[ncell,1], cells[ncell,2], neighbours[neighbour,0]],
                                          x[cells[ncell,0], cells[ncell,1], cells[ncell,2], neighbours[neighbour,0]],
                                          y[cells[ncell,0], cells[ncell,1], cells[ncell,2], neighbours[neighbour,0]],
                                          z[cells[ncell,0], cells[ncell,1], cells[ncell,2], neighbours[neighbour,0]],
                                          a[cell_x_target, cell_y_target, cell_z_target, neighbours[neighbour,1]],
                                          x[cell_x_target, cell_y_target, cell_z_target, neighbours[neighbour,1]],
                                          y[cell_x_target, cell_y_target, cell_z_target, neighbours[neighbour,1]],
                                          z[cell_x_target, cell_y_target, cell_z_target, neighbours[neighbour,1]],
                                          neighbours[neighbour,2], neighbours[neighbour,3], neighbours[neighbour,4])
        modifier.run(a, x, y, z)
        e1 = 0
        for target_number in range(number_of_targets):
            target = targets[target_number]
            neighbours = target.neighbours
            energy = target.energy
            number_of_neighbours = target.number_of_neighbours
            for ncell in range(number_of_cells):
                for neighbour in range(number_of_neighbours):
                    cell_x_target = (cells[ncell,0]+neighbours[neighbour,2]) % mod_x
                    cell_y_target = (cells[ncell,1]+neighbours[neighbour,3]) % mod_y
                    cell_z_target = (cells[ncell,2]+neighbours[neighbour,4]) % mod_z
                    e1 += energy.evaluate(a[cells[ncell,0], cells[ncell,1], cells[ncell,2], neighbours[neighbour,0]],
                                          x[cells[ncell,0], cells[ncell,1], cells[ncell,2], neighbours[neighbour,0]],
                                          y[cells[ncell,0], cells[ncell,1], cells[ncell,2], neighbours[neighbour,0]],
                                          z[cells[ncell,0], cells[ncell,1], cells[ncell,2], neighbours[neighbour,0]],
                                          a[cell_x_target, cell_y_target, cell_z_target, neighbours[neighbour,1]],
                                          x[cell_x_target, cell_y_target, cell_z_target, neighbours[neighbour,1]],
                                          y[cell_x_target, cell_y_target, cell_z_target, neighbours[neighbour,1]],
                                          z[cell_x_target, cell_y_target, cell_z_target, neighbours[neighbour,1]],
                                          neighbours[neighbour,2], neighbours[neighbour,3], neighbours[neighbour,4])
        de = e1-e0
        if not accept(de, temperature):
            not_accepted += 1
            modifier.undo_last_run(a, x, y, z)
    return iterations-not_accepted

@cython.cdivision(True)
cdef unsigned char accept(double dE, double kT):
    cdef double tmp
    if dE < 0:
        return True
    elif kT <= 0:
        return False
    else:
        return random() < exp(-dE/kT)
