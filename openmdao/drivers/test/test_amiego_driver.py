""" Testing the EGOLF driver."""

import unittest

from openmdao.api import IndepVarComp, Group, Problem, ExecComp
from openmdao.drivers.amiego_driver import AMIEGO_driver
from openmdao.test.branin import BranninInteger
from openmdao.test.util import assert_rel_error


class TestAMIEGOdriver(unittest.TestCase):

    def test_simple_brannin_opt(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('xC', 0.0), promotes=['*'])
        root.add('p2', IndepVarComp('xI', 0), promotes=['*'])
        root.add('comp', BranninInteger(), promotes=['*'])

        prob.driver = AMIEGO_driver()

        prob.driver.add_desvar('xI', lower=-5, upper=10)
        prob.driver.add_desvar('xC', lower=0.0, upper=15.0)

        prob.driver.add_objective('f')

        prob.setup(check=False)
        prob.run()

        # Optimal solution
        assert_rel_error(self, prob['f'], 2.38801229, 1e-5)

if __name__ == "__main__":
    unittest.main()
