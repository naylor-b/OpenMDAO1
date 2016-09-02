""" Testing the Branch and Bround driver."""

import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp
from openmdao.drivers.branch_and_bound import Branch_and_Bound
from openmdao.test.branin import BranninInteger
from openmdao.test.three_bar_truss import ThreeBarTruss
from openmdao.test.util import assert_rel_error


class TestBranchAndBounddriver(unittest.TestCase):

    def test_brannin_just_opt_integer(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p2', IndepVarComp('xI', 0), promotes=['*'])
        root.add('comp', BranninInteger(), promotes=['*'])

        prob.driver = Branch_and_Bound()
        prob.driver.options['use_surrogate'] = True
        prob.driver.options['disp'] = False

        prob.driver.add_desvar('xI', lower=-5, upper=10)
        prob.driver.add_objective('f')

        npt = 15
        prob.driver.sampling = {'xI' : np.linspace(0.0, 1.0, num=npt).reshape(npt, 1)}

        prob.setup(check=False)

        # Find an integer solution at a point where we know the minimum is in
        # the middle
        prob['xC'] = 13.0

        prob.run()

        # Optimal solution
        assert_rel_error(self, prob['xI'], -3, 1e-5)
        assert_rel_error(self, prob['f'], 1.62329296, 1e-5)

    def test_three_bar_truss_just_integer(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('xi_m1', IndepVarComp('mat1', 1), promotes=['*'])
        root.add('xi_m2', IndepVarComp('mat2', 1), promotes=['*'])
        root.add('xi_m3', IndepVarComp('mat3', 1), promotes=['*'])
        root.add('comp', ThreeBarTruss(), promotes=['*'])

        prob.driver = Branch_and_Bound()
        prob.driver.options['use_surrogate'] = True
        prob.driver.options['disp'] = False

        prob.driver.add_desvar('mat1', lower=1, upper=4)
        prob.driver.add_desvar('mat2', lower=1, upper=4)
        prob.driver.add_desvar('mat3', lower=1, upper=4)
        prob.driver.add_objective('mass')

        npt = 5
        samples = np.array([[1.0, 0.25, 0.75],
                            [0.0, 0.75, 0.0],
                            [0.75, 0.0, 0.25],
                            [0.75, 1.0, 0.5],
                            [0.25, 0.5, 1.0]]) 
        prob.driver.sampling = {'mat1' : samples[:, 0].reshape((npt, 1)),
                                'mat2' : samples[:, 1].reshape((npt, 1)),
                                'mat3' : samples[:, 2].reshape((npt, 1))}

        prob.setup(check=False)

        # Find an integer solution at the predetermined continuous optimum
        prob['area1'] = 5.1229065
        prob['area2'] = 4.14841357
        prob['area3'] = 1.00010045e-6

        prob.run()
        
        # Optimal solution
        assert_rel_error(self, prob['mat1'], 2, 1e-5)
        assert_rel_error(self, prob['mat2'], 2, 1e-5)
        assert_rel_error(self, prob['mat3'], 4, 1e-5)


if __name__ == "__main__":
    unittest.main()
