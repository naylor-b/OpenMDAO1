""" Testing for Problem.check_partial_derivatives."""

from six import iteritems
import unittest

import numpy as np

from openmdao.components.param_comp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.simple_comps import SimpleArrayComp, SimpleImplicitComp, \
                                      SimpleCompDerivMatVec
from openmdao.test.util import assert_rel_error


class TestProblemCheckPartials(unittest.TestCase):

    def test_double_diamond_model(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_array_model(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleArrayComp())
        prob.root.add('p1', ParamComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', ParamComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_bad_size(self):

        class BadComp(SimpleArrayComp):
            def jacobian(self, params, unknowns, resids):
                """Analytical derivatives"""
                J = {}
                J[('y', 'x')] = np.zeros((3, 3))
                return J


        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', BadComp())
        prob.root.add('p1', ParamComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        try:
            data = prob.check_partial_derivatives(out_stream=None)
        except Exception as err:
            msg = "Jacobian in component 'comp' between the" + \
                " variables 'x' and 'y' is the wrong size. " + \
                "It should be 2 by 2"
            self.assertEqual(str(err), msg)
        else:
            self.fail("Error expected")


class TestProblemFullFD(unittest.TestCase):

    def test_full_model_fd_simple_comp(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleCompDerivMatVec())
        prob.root.add('p1', ParamComp('x', 1.0))

        prob.root.connect('p1.x', 'comp.x')

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        param_list = ['comp.x']
        unknown_list = ['comp.y']

        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.y']['comp.x'][0][0], 2.0, 1e-6)


    def test_full_model_fd_simple_comp_promoted(self):

        prob = Problem()
        prob.root = Group()
        sub = prob.root.add('sub', Group(), promotes=['*'])
        sub.add('comp', SimpleCompDerivMatVec(), promotes=['*'])
        prob.root.add('p1', ParamComp('x', 1.0), promotes=['*'])

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        param_list = ['x']
        unknown_list = ['y']

        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_full_model_fd_double_diamond_grouped(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()
        prob.setup(check=False)
        prob.run()

        prob.root.fd_options['force_fd'] = True

        param_list = ['sub1.comp1.x1']
        unknown_list = ['comp7.y1']

        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['sub1.comp1.x1'][0][0], -40.75, 1e-6)

        prob.root.fd_options['form'] = 'central'
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['sub1.comp1.x1'][0][0], -40.75, 1e-6)


class TestProblemCheckTotals(unittest.TestCase):

    def test_double_diamond_model(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', ParamComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_full_model_fd_simple_comp_promoted(self):

        prob = Problem()
        prob.root = Group()
        sub = prob.root.add('sub', Group(), promotes=['*'])
        sub.add('comp', SimpleCompDerivMatVec(), promotes=['*'])
        prob.root.add('p1', ParamComp('x', 1.0), promotes=['*'])

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

if __name__ == "__main__":
    unittest.main()
