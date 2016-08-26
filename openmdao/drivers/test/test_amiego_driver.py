""" Testing the AMIEGO driver."""

import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp
from openmdao.drivers.amiego_driver import AMIEGO_driver
from openmdao.test.branin import BranninInteger
from openmdao.test.util import assert_rel_error


class TestAMIEGOdriver(unittest.TestCase):

    def test_simple_brannin_opt(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('xC', 7.5), promotes=['*'])
        root.add('p2', IndepVarComp('xI', 0), promotes=['*'])
        root.add('comp', BranninInteger(), promotes=['*'])

        prob.driver = AMIEGO_driver()
        prob.driver.cont_opt.options['tol'] = 1e-12
        root.deriv_options['type'] = 'fd'

        prob.driver.add_desvar('xI', lower=-5, upper=10)
        prob.driver.add_desvar('xC', lower=0.0, upper=15.0)

        prob.driver.add_objective('f')

        prob.driver.sampling = {'xI' : np.array([[0.0], [.76], [1.0]])}

        prob.setup(check=False)
        prob.run()

        # Optimal solution
        assert_rel_error(self, prob['f'], 0.49398, 1e-5)
        assert_rel_error(self, prob['xI'], -3.0, 1e-5)

if __name__ == "__main__":
    unittest.main()
