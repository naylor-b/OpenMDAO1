"""Testing ArrayCaseDriver"""

import unittest
from pprint import pformat
from types import GeneratorType

import numpy

from openmdao.api import IndepVarComp, Group, Problem, ExecComp

from openmdao.drivers.array_case_driver import ArrayCaseDriver

class DOEProblem(Problem):
    def __init__(self):
        super(DOEProblem, self).__init__()
        self.driver = driver = ArrayCaseDriver()

        root = self.root = Group()
        root.add('indep_var', IndepVarComp([('a', 0.5),('b',0.75),('c',0.9)]))
        root.add('comp', ExecComp(["x=a*2.0","y=b*3.0","z=c*1.5"]))

        root.connect('indep_var.a', 'comp.a')
        root.connect('indep_var.b', 'comp.b')
        root.connect('indep_var.c', 'comp.c')

        ncases = 30
        driver.desvar_array = numpy.arange(ncases*3,
                                           dtype=float).reshape(ncases, 3)

        driver.add_desvar('indep_var.a')
        driver.add_desvar('indep_var.b')
        driver.add_desvar('indep_var.c')

        driver.add_response(driver._desvars)
        driver.add_response(['comp.x', 'comp.y', 'comp.z'])


class TestArrayCaseDriver(unittest.TestCase):

    def test_array_case_driver(self):

        prob = DOEProblem()

        prob.setup(check=False)
        runList = prob.driver._build_runlist()
        prob.run()

        darr = prob.driver.desvar_array
        arr = prob.driver.response_array
        self.assertEqual(arr.shape, (30, 6))

        for i in range(arr.shape[0]):
            numpy.testing.assert_array_equal(arr[i,:],
                           numpy.array([
                              darr[i][0],
                              darr[i][1],
                              darr[i][2],
                              darr[i][0]*2.0,
                              darr[i][1]*3.0,
                              darr[i][2]*1.5,
                           ]))

if __name__ == "__main__":
    unittest.main()
