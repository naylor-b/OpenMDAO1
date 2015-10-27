
import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.drivers.fake_kona import FakeKona
from openmdao.test.util import assert_rel_error


class TestFakeKona(unittest.TestCase):
    def test_kona(self):
        prob = Problem(root=Group())
        root = prob.root

        P1 = root.add("P1", IndepVarComp("x", 1.0))

        G1 = root.add("G1", Group())
        C1 = G1.add("C1", ExecComp("y=2.0*x"))
        C2 = G1.add("C2", ExecComp("y=3.0*x1-x2"))

        G2 = root.add("G2", Group())
        C3 = G2.add("C3", ExecComp("y=4.0*x"))
        C4 = G2.add("C4", ExecComp("y=5.0*x1+x2"))

        root.connect("P1.x", ("G1.C1.x", "G1.C2.x1"))
        root.connect("G1.C2.y", "G2.C4.x1")
        root.connect("G1.C1.y", "G2.C3.x")
        G1.connect("C1.y", "C2.x2")
        G2.connect("C3.y", "C4.x2")

        prob.driver = FakeKona()
        prob.driver.add_desvar("P1.x")
        prob.driver.add_objective("G2.C4.y")
        prob.driver.add_constraint("G1.C2.y", upper=0.0)
        prob.driver.add_constraint("G2.C3.y", upper=0.0)

        prob.setup(check=False)

        prob.run()

        #print("obj")
        # set some uvec values and evaluate the objective
        prob.root.unknowns["G1.C2.y"] = 2.0
        prob.root.unknowns["G2.C3.y"] = 1.5

        prob.driver.quick_objective_eval()
        assert_rel_error(self, prob["G2.C4.y"], 11.5, 1e-6)

        #print("cons")
        # set some uvec values and evaluate the constraints
        prob.root.unknowns["P1.x"] = -1.0
        prob.root.unknowns["G1.C1.y"] = 3.0

        prob.driver.quick_constraint_eval()
        assert_rel_error(self, prob["G1.C2.y"], -6.0, 1e-6)
        assert_rel_error(self, prob["G2.C3.y"], 12.0, 1e-6)


if __name__ == '__main__':
    unittest.main()
