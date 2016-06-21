
import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, \
                         CaseDriver, InMemoryRecorder
from openmdao.test.exec_comp_for_test import ExecComp4Test
from openmdao.drivers.csv_case_driver import CSVCaseDriver, \
                                             save_cases_as_csv
from openmdao.test.util import ConcurrentTestCaseMixin

class TestCSVDriver(unittest.TestCase, ConcurrentTestCaseMixin):

    def setUp(self):
        self.concurrent_setUp(prefix='test_csv_drv-')

    def tearDown(self):
        self.concurrent_tearDown()

    def test_case_driver(self):
        problem = Problem()
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('indep_arr', IndepVarComp('a', val=np.zeros(4)))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', ExecComp4Test(["y=c*x","yarr=xarr*c"],
                                       xarr=np.zeros(4),
                                       yarr=np.zeros(4)))

        root.connect('indep_var.x', 'mult.x')
        root.connect('indep_arr.a', 'mult.xarr')
        root.connect('const.c', 'mult.c')

        cases = [
            [('indep_var.x', 3.0), ('const.c', 1.5),
                                   ('indep_arr.a',np.array([1.,2.,3.,4.]))],
            [('indep_var.x', 4.0), ('const.c', 2.),
                                   ('indep_arr.a',np.array([1.,2.,3.,4.]))],
            [('indep_var.x', 5.5), ('const.c', 4.0),
                                   ('indep_arr.a',np.array([1.,2.,3.,4.]))],
        ]

        cdriver = CaseDriver(cases)
        save_cases_as_csv(cdriver, "cases.csv")
        problem.driver = CSVCaseDriver("cases.csv")

        problem.driver.add_desvar('indep_var.x')
        problem.driver.add_desvar('indep_arr.a')
        problem.driver.add_desvar('const.c')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.setup(check=False)
        problem.run()

        for i, data in enumerate(problem.driver.recorders[0].iters):
            data['unknowns'] = dict(data['unknowns'])
            self.assertEqual(data['unknowns']['indep_var.x']*data['unknowns']['const.c'],
                             data['unknowns']['mult.y'])
            self.assertEqual(cases[i][0][1]*cases[i][1][1],
                             data['unknowns']['mult.y'])
            np.testing.assert_allclose(cases[i][2][1]*cases[i][1][1],
                                       data['unknowns']['mult.yarr'])

        self.assertEqual(len(problem.driver.recorders[0].iters), 3)

if __name__ == "__main__":
    unittest.main()
