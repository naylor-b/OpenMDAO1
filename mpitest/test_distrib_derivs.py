""" Test out some crucial linear GS tests in parallel with distributed comps."""

from __future__ import print_function

import unittest
import numpy

from openmdao.api import ParallelGroup, Group, Problem, IndepVarComp, \
    ExecComp, LinearGaussSeidel, Component
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error
from openmdao.util.array_util import evenly_distrib_idxs
from openmdao.core.mpi_wrap import MPI, under_mpirun, debug

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    rank = MPI.COMM_WORLD.rank
else:
    from openmdao.core.basic_impl import BasicImpl as impl
    rank = 0


class DistribExecComp(ExecComp):
    """An ExecComp that uses 2 procs and
    takes input var slices and has output var slices as well.
    """
    def __init__(self, exprs, arr_size=11, **kwargs):
        super(DistribExecComp, self).__init__(exprs, **kwargs)
        self.arr_size = arr_size

    def setup_distrib(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs. Returns a dict of
        index arrays keyed to variable names.
        """
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        for n, m in self._init_unknowns_dict.items():
            self.set_var_indices(n, val=numpy.ones(sizes[rank], float),
                                 src_indices=numpy.arange(start, end, dtype=int))

        for n, m in self._init_params_dict.items():
            self.set_var_indices(n, val=numpy.ones(sizes[rank], float),
                                 src_indices=numpy.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (2, 2)


class MPITests1(MPITestCase):

    N_PROCS = 1

    def test_too_few_procs(self):
        size = 3
        group = Group()
        group.add('P', IndepVarComp('x', numpy.ones(size)))
        group.add('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                        x=numpy.zeros(size),
                                        y=numpy.zeros(size)))
        group.add('C2', ExecComp(['z=3.0*y'],
                                 y=numpy.zeros(size),
                                 z=numpy.zeros(size)))

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.connect('P.x', 'C1.x')
        prob.root.connect('C1.y', 'C2.y')

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                             "This problem was given 1 MPI processes, "
                             "but it requires between 2 and 2.")
        else:
            if MPI:
                self.fail("Exception expected")


class MPITests2(MPITestCase):

    N_PROCS = 2

    def test_two_simple(self):
        size = 3
        group = Group()
        group.add('P', IndepVarComp('x', numpy.ones(size)))
        group.add('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                        x=numpy.zeros(size),
                                        y=numpy.zeros(size)))
        group.add('C2', ExecComp(['z=3.0*y'],
                                 y=numpy.zeros(size),
                                 z=numpy.zeros(size)))

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.connect('P.x', 'C1.x')
        prob.root.connect('C1.y', 'C2.y')

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['P.x'], ['C2.z'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['C2.z']['P.x'], numpy.eye(size)*6.0, 1e-6)

        J = prob.calc_gradient(['P.x'], ['C2.z'], mode='rev', return_format='dict')
        assert_rel_error(self, J['C2.z']['P.x'], numpy.eye(size)*6.0, 1e-6)

    def test_fan_out_grouped(self):
        size = 3
        prob = Problem(impl=impl)
        prob.root = root = Group()
        root.add('P', IndepVarComp('x', numpy.ones(size, dtype=float)))
        root.add('C1', DistribExecComp(['y=3.0*x'], arr_size=size,
                                       x=numpy.zeros(size, dtype=float),
                                       y=numpy.zeros(size, dtype=float)))
        sub = root.add('sub', ParallelGroup())
        sub.add('C2', ExecComp('y=1.5*x',
                               x=numpy.zeros(size),
                               y=numpy.zeros(size)))
        sub.add('C3', ExecComp(['y=5.0*x'],
                               x=numpy.zeros(size, dtype=float),
                               y=numpy.zeros(size, dtype=float)))

        root.add('C2', ExecComp(['y=x'],
                                x=numpy.zeros(size, dtype=float),
                                y=numpy.zeros(size, dtype=float)))
        root.add('C3', ExecComp(['y=x'],
                                x=numpy.zeros(size, dtype=float),
                                y=numpy.zeros(size, dtype=float)))
        root.connect('sub.C2.y', 'C2.x')
        root.connect('sub.C3.y', 'C3.x')

        root.connect("C1.y", "sub.C2.x")
        root.connect("C1.y", "sub.C3.x")
        root.connect("P.x", "C1.x")

        root.ln_solver = LinearGaussSeidel()
        root.sub.ln_solver = LinearGaussSeidel()

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['P.x'], ['C2.y', "C3.y"], mode='fwd', return_format='dict')
        assert_rel_error(self, J['C2.y']['P.x'], numpy.eye(size)*4.5, 1e-6)
        assert_rel_error(self, J['C3.y']['P.x'], numpy.eye(size)*15.0, 1e-6)

        J = prob.calc_gradient(['P.x'], ['C2.y', "C3.y"], mode='rev', return_format='dict')
        assert_rel_error(self, J['C2.y']['P.x'], numpy.eye(size)*4.5, 1e-6)
        assert_rel_error(self, J['C3.y']['P.x'], numpy.eye(size)*15.0, 1e-6)

    def test_fan_in_grouped(self):
        size = 3

        prob = Problem(impl=impl)
        prob.root = root = Group()

        root.add('P1', IndepVarComp('x', numpy.ones(size, dtype=float)))
        root.add('P2', IndepVarComp('x', numpy.ones(size, dtype=float)))
        sub = root.add('sub', ParallelGroup())

        sub.add('C1', ExecComp(['y=-2.0*x'],
                               x=numpy.zeros(size, dtype=float),
                               y=numpy.zeros(size, dtype=float)))
        sub.add('C2', ExecComp(['y=5.0*x'],
                               x=numpy.zeros(size, dtype=float),
                               y=numpy.zeros(size, dtype=float)))
        root.add('C3', DistribExecComp(['y=3.0*x1+7.0*x2'], arr_size=size,
                                       x1=numpy.zeros(size, dtype=float),
                                       x2=numpy.zeros(size, dtype=float),
                                       y=numpy.zeros(size, dtype=float)))
        root.add('C4', ExecComp(['y=x'],
                                x=numpy.zeros(size, dtype=float),
                                y=numpy.zeros(size, dtype=float)))

        root.connect("sub.C1.y", "C3.x1")
        root.connect("sub.C2.y", "C3.x2")
        root.connect("P1.x", "sub.C1.x")
        root.connect("P2.x", "sub.C2.x")
        root.connect("C3.y", "C4.x")

        root.ln_solver = LinearGaussSeidel()
        root.sub.ln_solver = LinearGaussSeidel()

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['P1.x', 'P2.x'], ['C4.y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['C4.y']['P1.x'], numpy.eye(size)*-6.0, 1e-6)
        assert_rel_error(self, J['C4.y']['P2.x'], numpy.eye(size)*35.0, 1e-6)

        J = prob.calc_gradient(['P1.x', 'P2.x'], ['C4.y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['C4.y']['P1.x'], numpy.eye(size)*-6.0, 1e-6)
        assert_rel_error(self, J['C4.y']['P2.x'], numpy.eye(size)*35.0, 1e-6)

    def test_src_indices_error(self):
        size = 3
        group = Group()
        group.add('P', IndepVarComp('x', numpy.ones(size)))
        group.add('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                        x=numpy.zeros(size),
                                        y=numpy.zeros(size)))
        group.add('C2', ExecComp(['z=3.0*y'],
                                 y=numpy.zeros(size),
                                 z=numpy.zeros(size)))

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.connect('P.x', 'C1.x')
        prob.root.connect('C1.y', 'C2.y')

        prob.driver.add_desvar('P.x')
        prob.driver.add_objective('C1.y')

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err), "'C1.y' is a distributed variable"
                                       " and may not be used as a design var,"
                                       " objective, or constraint.")
        else:
            if MPI:
                self.fail("Exception expected")

class DistribEvenOddComp(Component):
    """Uses 2 procs and takes input var slices"""
    def __init__(self, arr_size=11, numprocs=2):
        super(DistribEvenOddComp, self).__init__()
        self.arr_size = arr_size
        self.num_procs = numprocs
        self.add_param('x', numpy.ones(arr_size, float))
        self.add_output('y', numpy.ones(arr_size, float))

    def solve_nonlinear(self, params, unknowns, resids):
        if MPI:
            rank = self.comm.rank
            offset = self.offsets[rank]
            size = self.sizes[rank]

            # even ranks
            if rank % 2 == 0:
                unknowns['y'] = 2.0 * params['x']
            else:
                unknowns['y'] = 3.0 * params['x']
        else:
            unknowns['y'] = params['x'] * 2.0

    def setup_distrib(self):
        rank = self.comm.rank

        self.sizes, self.offsets = evenly_distrib_idxs(self.comm.size,
                                                       self.arr_size)
        start = self.offsets[rank]
        end = start + self.sizes[rank]

        #need to initialize the param to have the correct local size
        self.set_var_indices('x', val=numpy.ones(self.sizes[rank], float),
                             src_indices=numpy.arange(start, end, dtype=int))
        self.set_var_indices('y', val=numpy.ones(self.sizes[rank], float),
                             src_indices=numpy.arange(start, end, dtype=int))

    def get_req_procs(self):
        return (self.num_procs, self.num_procs)

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives"""
        J = {}

        if MPI:
            rank = self.comm.rank
            if rank % 2 == 0:
                J[('y', 'x')] = numpy.eye(self.sizes[rank]) * 2.0
            else:
                J[('y', 'x')] = numpy.eye(self.sizes[rank]) * 3.0
        else:
            J[('y', 'x')] = numpy.eye(self.arr_size) * 2.0

        #debug("%s linearize J:"%self.pathname,J)
        return J

class TestParallelDerivs(MPITestCase):
    N_PROCS = 2

    def test_par_deriv_distrib(self):
        size = 4
        group = Group()
        group.add('P', IndepVarComp('x', numpy.ones(size)))
        C1 = group.add('C1', DistribEvenOddComp(arr_size=size, numprocs=self.N_PROCS))
        C1._striped = True
        # group.add('C2', ExecComp(['y=3.0*x'],
        #                          y=numpy.zeros(size),
        #                          x=numpy.zeros(size)))

        prob = Problem(impl=impl, root=group)

        prob.root.ln_solver.options['mode'] = 'rev'

        prob.root.connect('P.x', 'C1.x')
        #prob.root.connect('C1.y', 'C2.x')

        prob.driver.add_desvar('P.x')
        prob.driver.add_objective('C1.y')

        prob.driver.parallel_derivs([('C1.y',2)])

        # import sys
        # sys.path.insert(0, ".")
        # import wingdbstub

        prob.setup(check=False)
        prob.run()

        # debug("P.x:",prob['P.x'])
        # debug('C1.x:',prob['C1.x'])
        # debug('C1.y:',prob['C1.y'])
        #
        # debug("objs:",prob.driver.get_objectives())
        #
        # debug("========================================================")
        J = prob.calc_gradient(['P.x'], ['C1.y'], mode='rev', return_format='dict')
        #debug("J",J)
        if MPI:
            rank = self.comm.rank
            vals = [2.0, 3.0]
            expected = numpy.zeros(size)
            for i in range(self.comm.size):
                start = C1.offsets[i]
                sz = C1.sizes[i]
                expected[start:start+sz] = vals[i]
            expected = numpy.diag(expected)

            assert_rel_error(self, J['C1.y']['P.x'], expected, 1e-6)
        else:
            assert_rel_error(self, J['C1.y']['P.x'], numpy.eye(size)*2.0, 1e-6)


#class DistComp(Component):
    #"""Uses 2 procs and has output var slices"""
    #def __init__(self, arr_size=4):
        #super(DistComp, self).__init__()
        #self.arr_size = arr_size
        #self.add_param('invec', numpy.ones(arr_size, float))
        #self.add_output('outvec', numpy.ones(arr_size, float))

    #def solve_nonlinear(self, params, unknowns, resids):

        #p1 = params['invec'][0]
        #p2 = params['invec'][1]

        #unknowns['outvec'][0] = p1**2 - 11.0*p2
        #unknowns['outvec'][1] = 7.0*p2**2 - 13.0*p1

    #def linearize(self, params, unknowns, resids):
        #""" Derivatives"""

        #p1 = params['invec'][0]
        #p2 = params['invec'][1]

        #J = {}
        #jac = numpy.zeros((2, 2))
        #jac[0][0] = 2.0*p1
        #jac[0][1] = -11.0
        #jac[1][0] = -13.0
        #jac[1][1] = 7.0*p2

        #J[('outvec', 'invec')] = jac
        #return J

    #def setup_distrib(self):
        #""" component declares the local sizes and sets initial values
        #for all distributed inputs and outputs. Returns a dict of
        #index arrays keyed to variable names.
        #"""

        #comm = self.comm
        #rank = comm.rank

        #sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        #start = offsets[rank]
        #end = start + sizes[rank]

        #self.set_var_indices('invec', val=numpy.ones(sizes[rank], float),
                             #src_indices=numpy.arange(start, end, dtype=int))
        #self.set_var_indices('outvec', val=numpy.ones(sizes[rank], float),
                             #src_indices=numpy.arange(start, end, dtype=int))

    #def get_req_procs(self):
        #return (2, 2)

#class ElementwiseParallelDerivativesTestCase(MPITestCase):

    #N_PROCS = 2

    #def test_simple_adjoint(self):
        #if not MPI:
            #raise unittest.SkipTest("this test only works in MPI")
        #top = Problem(impl=impl)
        #root = top.root = Group()
        #root.add('p1', IndepVarComp('x', numpy.ones((4, ))))
        #root.add('dcomp', DistComp(arr_size=4))

        #top.driver.add_desvar('p1.x', numpy.ones((4, )))
        #top.driver.add_objective('dcomp.outvec')
        #top.root.connect('p1.x', 'dcomp.invec')

        #top.setup(check=False)

        #top['p1.x'][0] = 1.0
        #top['p1.x'][1] = 2.0
        #top['p1.x'][2] = 3.0
        #top['p1.x'][3] = 4.0

        #top.run()

        #J = top.calc_gradient(['p1.x'], ['dcomp.outvec'], mode='rev')

        #print("J:",J)
        #assert_rel_error(self, J[0][0], 2.0, 1e-6)
        #assert_rel_error(self, J[0][1], -11.0, 1e-6)
        #assert_rel_error(self, J[0][2], 4.0, 1e-6)
        #assert_rel_error(self, J[0][3], -11.0, 1e-6)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
