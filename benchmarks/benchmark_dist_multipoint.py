from __future__ import print_function
import unittest
from six.moves import range
import numpy as np

import time
from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp


class Plus(Component):
    def __init__(self):
        super(Plus, self).__init__()
        self.add_param('x', 0.)
        self.add_param('adder', 0.)
        self.add_output('f1', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['f1'] = params['x'] + params['adder']

class Times(Component):
    def __init__(self):
        super(Times, self).__init__()
        self.add_param('f1', 0.)
        self.add_param('scaler', 1.0)
        self.add_output('f2', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['f2'] = params['f1'] + params['scaler']

class Point(Group):

    def __init__(self):
        super(Point, self).__init__()
        self.add('plus', Plus(), promotes=['*'])
        self.add('times', Times(), promotes=['*'])

class Summer(Component):

    def __init__(self, size):
        super(Summer, self).__init__()
        self.add_param('y', 0.)

        self.add_output('total', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['total'] = np.sum(params["y"])

class BM(unittest.TestCase):

    def _setup_bm(self, npts):

        prob = Problem(root=Group())
        prob.root.add("indep",
                      IndepVarComp([
                          ("s", np.zeros(npts)),
                          ("a", np.zeros(npts)),
                      ]))
        multipoint = prob.root.add("multi", Group(num_multipoints=npts))
        multipoint.add("point", Point())

        prob.root.add("aggregate", Summer(npts))

        prob.root.connect("indep.a", "multi.point.plus.adder")
        prob.root.connect("indep.s", "multi.point.times.scaler")
        prob.root.connect("multi.point.times.f2", "aggregate.y")

        prob.setup(check=False)

        return prob

    # def benchmark_run_5K(self):
    #     p = self._setup_bm(5000)
    #     p.run()
    #
    # def benchmark_run_2K(self):
    #     p = self._setup_bm(2000)
    #     p.run()

    def benchmark_run_1K(self):
        p = self._setup_bm(1000)
        p.run()
