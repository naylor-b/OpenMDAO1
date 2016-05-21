
from __future__ import print_function

import time
import traceback
import numpy

from openmdao.api import IndepVarComp, Component, Group, Problem, \
                         ArrayCaseDriver, InMemoryRecorder
from openmdao.test.exec_comp_for_test import ExecComp4Test
from openmdao.core.mpi_wrap import MPI, debug, MultiProcFailCheck
from openmdao.test.mpi_util import MPITestCase

if MPI: # pragma: no cover
    # if you called this script with 'mpirun', then use the petsc data passing
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    # if you didn't use `mpirun`, then use the numpy data passing
    from openmdao.api import BasicImpl as impl


class PDRTestCase(MPITestCase):

    N_PROCS = 4

    def test_response(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp([('a', 0.5),('b',0.75),('c',0.9)]))
        root.add('comp', ExecComp4Test(["x=a*2.0","y=b*3.0","z=c*1.5"]))

        root.connect('indep_var.a', 'comp.a')
        root.connect('indep_var.b', 'comp.b')
        root.connect('indep_var.c', 'comp.c')

        driver = ArrayCaseDriver(num_par_doe=self.N_PROCS)
        problem.driver = driver

        ncases = 30
        driver.desvar_array = numpy.arange(ncases*3, dtype=float).reshape(ncases, 3)

        print(driver.desvar_array)
        print('----')

        driver.add_desvar('indep_var.a')
        driver.add_desvar('indep_var.b')
        driver.add_desvar('indep_var.c')

        driver.add_response(driver._desvars)
        driver.add_response(['comp.x', 'comp.y', 'comp.z'])

        problem.setup(check=False)
        problem.run()

        if self.comm.rank == 0:
            arr = driver.response_array

            print(arr)

            self.assertEqual(arr.shape, (ncases, 6))
            for i in range(ncases):
                self.assertEqual(arr[i,0]*2.0, arr[i,3])
                self.assertEqual(arr[i,1]*3.0, arr[i,4])
                self.assertEqual(arr[i,2]*1.5, arr[i,5])

        # test re-run
        ncases = 27
        driver.desvar_array = numpy.arange(ncases*3, dtype=float).reshape(ncases, 3)
        problem.run()

        if self.comm.rank == 0:
            arr = driver.response_array
            self.assertEqual(arr.shape, (ncases, 6))
            for i in range(ncases):
                self.assertEqual(arr[i,0]*2.0, arr[i,3])
                self.assertEqual(arr[i,1]*3.0, arr[i,4])
                self.assertEqual(arr[i,2]*1.5, arr[i,5])

    def test_lb_response(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp([('a', 0.5),('b',0.75),('c',0.9)]))
        root.add('comp', ExecComp4Test(["x=a*2.0","y=b*3.0","z=c*1.5"]))

        root.connect('indep_var.a', 'comp.a')
        root.connect('indep_var.b', 'comp.b')
        root.connect('indep_var.c', 'comp.c')

        driver = ArrayCaseDriver(num_par_doe=self.N_PROCS, load_balance=True)
        problem.driver = driver

        ncases = 30
        driver.desvar_array = numpy.arange(ncases*3, dtype=float).reshape(ncases, 3)

        driver.add_desvar('indep_var.a')
        driver.add_desvar('indep_var.b')
        driver.add_desvar('indep_var.c')

        driver.add_response(driver._desvars)
        driver.add_response(['comp.x', 'comp.y', 'comp.z'])

        problem.setup(check=False)
        problem.run()

        if self.comm.rank == 0:
            arr = driver.response_array
            self.assertEqual(arr.shape, (ncases, 6))
            for i in range(ncases):
                self.assertEqual(arr[i,0]*2.0, arr[i,3])
                self.assertEqual(arr[i,1]*3.0, arr[i,4])
                self.assertEqual(arr[i,2]*1.5, arr[i,5])

        # test re-run
        ncases = 27
        driver.desvar_array = numpy.arange(ncases*3, dtype=float).reshape(ncases, 3)
        problem.run()

        if self.comm.rank == 0:
            arr = driver.response_array
            self.assertEqual(arr.shape, (ncases, 6))
            for i in range(ncases):
                self.assertEqual(arr[i,0]*2.0, arr[i,3])
                self.assertEqual(arr[i,1]*3.0, arr[i,4])
                self.assertEqual(arr[i,2]*1.5, arr[i,5])

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
