""" Driver for AMIEGO (A Mixed Integer Efficient Global Optimization).

This driver is based on the EGO-Like Framework (EGOLF) for the simultaneous
design-mission-allocation optimization problem. Handles
mixed-integer/discrete type design variables in a computationally efficient
manner and finds a near-global solution to the above MINLP/MDNLP problem.

Developed by Satadru Roy
Purdue University, West Lafayette, IN
Copyright, July 2016
Implemented in OpenMDAO, Aug 2016, Kenneth T. Moore
"""

import numpy as np

from openmdao.core.driver import Driver
from openmdao.drivers.scipy_optimizer import ScipyOptimizer


class AMIEGO_driver(Driver):
    """ Driver for AMIEGO (A Mixed Integer Efficient Global Optimization).
    This driver is based on the EGO-Like Framework (EGOLF) for the
    simultaneous design-mission-allocation optimization problem. It handles
    mixed-integer/discrete type design variables in a computationally
    efficient manner and finds a near-global solution to the above
    MINLP/MDNLP problem. The continuous optimization is handled by the
    optimizer slotted in self.cont_opt.

    EGOLF_driver supports the following:
        integer_design_vars

    Options
    -------
    options['ei_tol_rel'] :  0.001
        Relative tolerance on the expected improvement.
    options['ei_tol_abs'] :  0.001
        Absolute tolerance on the expected improvement.
    """

    def __init__(self):
        """Initialize the AMIEGO driver."""

        super(AMIEGO_driver, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = True
        self.supports['mixed_integer'] = True

        # Options
        opt = self.options
        opt.add_option('ei_tol_rel', 0.001, lower=0.0,
                       desc='Relative tolerance on the expected improvement.')
        opt.add_option('ei_tol_abs', 0.001, lower=0.0,
                       desc='Absolute tolerance on the expected improvement.')

        # The default continuous optimizer.
        self.cont_opt = ScipyOptimizer()
        self.cont_opt.options['optimizer'] = 'SLSQP'

        self._obj_metamodel = None
        self._con_metamodel = {}

    def _setup(self):
        """  Initialize whatever we need."""
        super(AMIEGO_driver, self)._setup()
        self.cont_opt._setup()

        # These should be perfectly okay to 'share' with the continuous
        # sub-optimizer.
        self.cont_opt._cons = self._cons
        self.cont_opt._objs = self._objs

    def set_root(self, pathname, root):
        """ Sets the root Group of this driver.

        Args
        ----
        root : Group
            Our root Group.
        """
        super(AMIEGO_driver, self).set_root(pathname, root)
        self.cont_opt.set_root(pathname, root)

    def run(self, problem):
        """pyOpt execution. Note that pyOpt controls the execution, and the
        individual optimizers (i.e., SNOPT) control the iteration.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """

        # Step 2: Perform the optimization w.r.t continuous design variables
        self.cont_opt.run(problem)