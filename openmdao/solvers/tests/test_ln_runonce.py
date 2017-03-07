"""Test the LNRunOnce linear solver class."""

import unittest

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.devtools.testutil import assert_rel_error
from openmdao.solvers.ln_runonce import LNRunOnce
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.groups.parallel_groups import ConvergeDivergeGroups


class TestLNRunOnceSolver(unittest.TestCase):

    def test_converge_diverge_groups(self):
        # Test derivatives for converge-diverge-groups topology.
        prob = Problem()
        prob.model = ConvergeDivergeGroups()

        prob.model.ln_solver = LNRunOnce()
        prob.model.suppress_solver_output = True

        g1 = prob.model.get_subsystem('g1')
        g2 = g1.get_subsystem('g2')
        g3 = prob.model.get_subsystem('g3')
        g1.ln_solver = LNRunOnce()
        g2.ln_solver = LNRunOnce()
        g3.ln_solver = LNRunOnce()
        g1.suppress_solver_output = True
        g2.suppress_solver_output = True
        g3.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['iv.x']
        of = ['c7.y1']

        # Make sure value is fine.
        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

    def test_feature_solver(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.ln_solver = LNRunOnce()

        prob.setup(check=False, mode='fwd')

        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, derivs['f_xy']['x'], -6.0, 1e-6)
        assert_rel_error(self, derivs['f_xy']['y'], 8.0, 1e-6)

if __name__ == "__main__":
    unittest.main()
