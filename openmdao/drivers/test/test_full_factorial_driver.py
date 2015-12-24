"""Testing FullFactorialDriver"""

import unittest
from pprint import pformat
from types import GeneratorType

from openmdao.api import IndepVarComp, Group, Problem
from openmdao.test.paraboloid import Paraboloid

from openmdao.drivers.fullfactorial_driver import FullFactorialDriver

class TestFullFactorial(unittest.TestCase):

    def test_fullfactorial(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = FullFactorialDriver(2)
        prob.driver.add_desvar('x', lower=0, upper=1)
        prob.driver.add_desvar('y', lower=0, upper=1)
        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        runList = prob.driver._build_runlist()
        prob.run()

        # Assert that the runList generated is of Generator Type
        self.assertTrue((type(runList) == GeneratorType),
                        "_build_runlist did not return a generator.")

        # Assert that the cases generated by the driver are correct
        inputs = set()
        for case in runList:
            case = dict(case)
            inputs.add((case['x'], case['y']))

        self.assertTrue(len(inputs) == 4,
                         "Incorrect number of runs generated.")

        correctInputs = {(0,0), (0,1), (1,0), (1,1)}

        self.assertTrue(inputs == correctInputs,
                        "Incorrect inputs generated.")

if __name__ == "__main__":
    unittest.main()
    
