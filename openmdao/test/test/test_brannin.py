""" Testing the simple Branin problem."""

import unittest

from openmdao.api import IndepVarComp, Group, Problem, ScipyOptimizer, ExecComp
from openmdao.test.branin import Branin
from openmdao.test.util import assert_rel_error


class TestBranin(unittest.TestCase):

    def test_simple_branin_opt(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x0', 0.0), promotes=['*'])
        root.add('p2', IndepVarComp('x1', 0.0), promotes=['*'])
        root.add('comp', Branin(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('x0', lower=-5.0, upper=10.0)
        prob.driver.add_desvar('x1', lower=0.0, upper=15.0)
        prob.driver.options['disp'] = False

        prob.driver.add_objective('f')

        prob.setup(check=False)
        prob.run()

        # Optimal solution
        assert_rel_error(self, prob['f'], 0.397887, 1e-5)

if __name__ == "__main__":
    unittest.main()
