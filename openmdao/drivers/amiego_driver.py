""" Driver for AMIEGO (A Mixed Integer Efficient Global Optimization).

This driver is based on the EGO-Like Framework (EGOLF) for the simultaneous
design-mission-allocation optimization problem. Handles
mixed-integer/discrete type design variables in a computationally efficient
manner and finds a near-global solution to the above MINLP/MDNLP problem.

Developed by Satadru Roy
Purdue University, West Lafayette, IN
July 2016
Implemented in OpenMDAO, Aug 2016, Kenneth T. Moore
"""

from __future__ import print_function
from copy import deepcopy

from six import iteritems
from six.moves import range

import numpy as np

from openmdao.core.driver import Driver
from openmdao.core.vec_wrapper import _ByObjWrapper
from openmdao.drivers.scipy_optimizer import ScipyOptimizer
from openmdao.surrogate_models.kriging import KrigingSurrogate
from openmdao.util.record_util import create_local_meta, update_local_meta


class AMIEGO_driver(Driver):
    """ Driver for AMIEGO (A Mixed Integer Efficient Global Optimization).
    This driver is based on the EGO-Like Framework (EGOLF) for the
    simultaneous design-mission-allocation optimization problem. It handles
    mixed-integer/discrete type design variables in a computationally
    efficient manner and finds a near-global solution to the above
    MINLP/MDNLP problem. The continuous optimization is handled by the
    optimizer slotted in self.cont_opt.

    AMIEGO_driver supports the following:
        integer_design_vars

    Options
    -------
    options['ei_tol_rel'] :  0.001
        Relative tolerance on the expected improvement.
    options['ei_tol_abs'] :  0.001
        Absolute tolerance on the expected improvement.
    options['max_infill_points'] : 10
        Ratio of maximum number of additional points to number of initial
        points.

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
        opt.add_option('max_infill_points', 10, lower=1.0,
                       desc='Ratio of maximum number of additional points to number of initial points.')

        # The default continuous optimizer. User can slot a different one
        self.cont_opt = ScipyOptimizer()
        self.cont_opt.options['optimizer'] = 'SLSQP'

        # Default surrogate. User can slot a modified one, but it essentially
        # has to provide what Kriging provides.
        self.surrogate = KrigingSurrogate

        self.c_dvs = []
        self.i_dvs = []
        self.i_size = 0
        self.idx_cache = {}
        self.record_name = 'AMIEGO'

        # Initial Sampling
        # TODO: Somehow slot an object that generates this (LHC for example)
        self.sampling = {}

    def _setup(self):
        """  Initialize whatever we need."""
        super(AMIEGO_driver, self)._setup()
        cont_opt = self.cont_opt
        cont_opt._setup()
        cont_opt.record_name = self.record_name + cont_opt.record_name

        # Identify and size our design variables.
        self.i_size = 0
        j = 0
        for name, val in iteritems(self.get_desvars()):
            if isinstance(val, _ByObjWrapper):
                self.i_dvs.append(name)
                try:
                    self.i_size += len(np.asarray(val.val))
                except TypeError:
                    self.i_size += 1
                self.idx_cache[name] = (j, j+self.i_size)
                j += self.i_size
            else:
                self.c_dvs.append(name)

        # Continuous Optimization only gets continuous desvars
        for name in self.c_dvs:
            cont_opt._desvars[name] = self._desvars[name]

        # It should be perfectly okay to 'share' obj and con with the
        # continuous sub-optimizer.
        cont_opt._cons = self._cons
        cont_opt._objs = self._objs

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
        """Execute the AMIEGO driver.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """
        n_i = self.i_size
        ei_tol_rel = self.options['ei_tol_rel']
        ei_tol_abs = self.options['ei_tol_abs']
        cont_opt = self.cont_opt

        # Metadata Setup
        self.metadata = create_local_meta(None, self.record_name)
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count, ))

        #----------------------------------------------------------------------
        # Step 1: Generate a set of initial integer points
        # TODO: Use Latin Hypercube Sampling to generate the initial points
        # User supplied (in future use LHS). Provide num_xI+2 starting points
        #----------------------------------------------------------------------

        n_train = self.sampling[self.i_dvs[0]].shape[0]
        max_pt_lim = self.options['max_infill_points']*n_train

        # Since we need to add a new point every iteration, make these lists
        # for speed.
        x_i = []
        obj = []
        c_start = 0
        c_end = n_train

        for i_train in range(n_train):
            for var in self.i_dvs:
                lower = self._desvars[var]['lower']
                upper = self._desvars[var]['upper']
                i, j = self.idx_cache[var]
                x_i_0 = self.sampling[var][i_train, :]

                xx_i = np.round(lower + x_i_0 * (upper - lower))
            x_i.append(xx_i)

        ei_max = 1.0
        term = 0.0
        terminate = False
        tot_newpt_added = 0
        best_obj = 1.0e99

        # AMIEGO main loop
        while ei_max>term and terminate==False and tot_newpt_added<max_pt_lim:
            self.iter_count += 1

            #------------------------------------------------------------------
            # Step 2: Perform the optimization w.r.t continuous design
            # variables
            #------------------------------------------------------------------

            for i_run in range(c_start, c_end):

                # Set Integer design variables
                for var in self.i_dvs:
                    i, j = self.idx_cache[var]
                    self.set_desvar(var, x_i[i_run][i:j])

                # Optimize continuous variables
                cont_opt.run(problem)

                # Get objectives and constraints (TODO)
                current_objs = self.get_objectives()
                obj_name = list(current_objs.keys())[0]
                current_obj = current_objs[obj_name].copy()
                obj.append(current_obj)

                print(self.get_desvars())
                print(obj[i_run])

                # If best solution, save it
                if current_obj < best_obj:
                    best_obj = current_obj

                    # Save integer and continuous DV
                    # NOTE: is deepcopying the dictionary slow?
                    best_design = deepcopy(self.get_desvars())

            #------------------------------------------------------------------
            # Step 3: Build the surrogate models
            #------------------------------------------------------------------

            obj_surrogate = self.surrogate()
            obj_surrogate.train(x_i, obj)

            #------------------------------------------------------------------
            # Step 4: Maximize the expected improvement function to obtain an
            # integer infill point.
            #------------------------------------------------------------------


            #------------------------------------------------------------------
            # Step 5: Check for termination
            #------------------------------------------------------------------

            # JUST ONE FOR NOW
            terminate = True

            c_start = c_end
            c_end += 1

            if np.abs(best_obj)<= 1e-6:
                term = ei_tol_abs
            else:
                term = np.min(np.array([np.abs(ei_tol_rel*best_obj), ei_tol_abs]))

        if ei_max <= term:
            print("No Further improvement expected! Terminating algorithm.")
        elif terminate:
            print("No new point found that improves the surrogate. Terminating algorithm.")
        elif Tot_newpt_added >= max_pt_lim:
            print("Maximum allowed sampling limit reached! Terminating algorithm.")

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name, val in iteritems(best_design):
            if isinstance(val, _ByObjWrapper):
                self.set_desvar(name, val.val)
            else:
                self.set_desvar(name, val)

        with self.root._dircontext:
            self.root.solve_nonlinear(metadata=self.metadata)

