""" Class definition for the Branch_and_Bound driver. This driver can be run
standalone or plugged into the AMIEGO driver.

This is the branch and bound algorithm that maximizes the constrained
expected improvement function and returns an integer infill point. The
algorithm uses the relaxation techniques proposed by Jones et.al. on their
paper on EGO,1998. This enables the algorithm to use any gradient-based
approach to obtain a global solution. Also, to satisfy the integer
constraints, a new branching scheme has been implemented.

Developed by Satadru Roy
School of Aeronautics & Astronautics
Purdue University, West Lafayette, IN 47906
July, 2016
Implemented in OpenMDAO, Aug 2016, Kenneth T. Moore
"""

from __future__ import print_function

from six import iteritems
from six.moves import range

import numpy as np

from openmdao.core.driver import Driver
from openmdao.surrogate_models.kriging import KrigingSurrogate
from openmdao.util.record_util import create_local_meta, update_local_meta


class Branch_and_Bound(Driver):
    """ Class definition for the Branch_and_Bound driver. This driver can be run
    standalone or plugged into the AMIEGO driver.

    This is the branch and bound algorithm that maximizes the constrained
    expected improvement function and returns an integer infill point. The
    algorithm uses the relaxation techniques proposed by Jones et.al. on
    their paper on EGO,1998. This enables the algorithm to use any
    gradient-based approach to obtain a global solution. Also, to satisfy the
    integer constraints, a new branching scheme has been implemented.
    """

    def __init__(self):
        """Initialize the Branch_and_Bound driver."""

        super(Branch_and_Bound, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = False
        self.supports['mixed_integer'] = False

        # Default surrogate. User can slot a modified one, but it essentially
        # has to provide what Kriging provides.
        self.surrogate = KrigingSurrogate

        # Options
        opt = self.options
        opt.add_option('atol', 1.0e-6, lower=0.0,
                       desc='Absolute tolerance of upper minus lower bound '
                       'for termination.')
        opt.add_option('concave_EI', False,
                       desc='Set to True to apply a transformation to make the '
                       'objective function concave.')
        opt.add_option('integer_tol', 1.0e-6, lower=0.0,
                       desc='Integer Rounding Tolerance.')
        opt.add_option('use_surrogate', False,
                       desc='Use surrogate model for the optimization. Training '
                       'data must be supplied.')

        # Initial Sampling
        # TODO: Somehow slot an object that generates this (LHC for example)
        self.sampling = {}

        self.dvs = []
        self.size = 0
        self.idx_cache = {}
        self.obj_surrogate = None
        self.record_name = 'B&B'
        self.con_cache = None

        # When this is slotted into AMIEGO, this will be set to False.
        self.standalone = True

    def _setup(self):
        """  Initialize whatever we need."""
        super(Branch_and_Bound, self)._setup()

        # Size our design variables.
        j = 0
        for name, val in iteritems(self.get_desvars()):
            self.dvs.append(name)
            try:
                self.size += len(np.asarray(val.val))
            except TypeError:
                self.size += 1
                self.idx_cache[name] = (j, j+self.size)
                j += self.size

        # Lower and Upper bounds
        self.lb = np.empty((self.size, ))
        self.ub = np.empty((self.size, ))
        dv_dict = self._desvars
        for var in self.dvs:
            i, j = self.idx_cache[var]
            self.lb[i:j] = dv_dict[var]['lower']
            self.ub[i:j] = dv_dict[var]['upper']

    def run(self, problem):
        """Execute the Branch_and_Bound method.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """
        obj_surrogate = self.obj_surrogate

        # Metadata Setup
        self.metadata = create_local_meta(None, self.record_name)
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count, ))

        # Use surrogates for speed.
        # Note, when not standalone, the parent driver provides the surrogate
        # model.
        if self.standalone and self.options['use_surrogate']:

            n_train = self.sampling[self.dvs[0]].shape[0]
            x_i = []
            obj = []
            system = self.root
            metadata = self.metadata

            for i_train in range(n_train):
                for var in self.dvs:
                    lower = self._desvars[var]['lower']
                    upper = self._desvars[var]['upper']
                    i, j = self.idx_cache[var]
                    x_i_0 = self.sampling[var][i_train, :]

                    xx_i = np.round(lower + x_i_0 * (upper - lower))
                x_i.append(xx_i)

            # Run each case and extract obj/con
            for i_run in range(len(x_i)):

                # Set design variables
                for var in self.dvs:
                    i, j = self.idx_cache[var]
                    self.set_desvar(var, x_i[i_run][i:j])

                with system._dircontext:
                    system.solve_nonlinear(metadata=metadata)

                # Get objectives and constraints (TODO)
                current_objs = self.get_objectives()
                obj_name = list(current_objs.keys())[0]
                current_obj = current_objs[obj_name].copy()
                obj.append(current_obj)

            obj_surrogate = self.surrogate()
            obj_surrogate.train(x_i, obj)

        # Calculate intermediate statistics
        if obj_surrogate:
            #obj_surrogate.X = surrogate.X
            #obj_surrogate.ynorm = surrogate.Y
            #obj_surrogate.thetas = surrogate.thetas
            obj_surrogate.mu = np.mean(obj_surrogate.Y) #This value should always be 0.0
            obj_surrogate.SigmaSqr = obj_surrogate.sigma2/np.square(obj_surrogate.Y_std) #This value should always be 1.0
            #obj_surrogate.c_r = obj_surrogate.alpha
            # TODO: mvp in Kriging
            obj_surrogate.R_inv = obj_surrogate.Vh.T.dot(np.einsum('i,ij->ij', obj_surrogate.S_inv, obj_surrogate.U.T))
            #obj_surrogate.X_std = obj_surrogate.X_std.reshape(num_xI,1)
            #obj_surrogate.X_mean = obj_surrogate.X_mean.reshape(num_xI,1)

        #----------------------------------------------------------------------
        # Step 1: Initialize
        #----------------------------------------------------------------------

        terminate = False
        num_des = len(self.xI_lb)

        # Initial B&B bounds are infinite.
        LBD = -np.inf
        UBD = np.inf

        xL_iter = self.xI_lb.copy()
        xU_iter = self.xI_ub.copy()

        # Active set fields:
        #     Aset = [[NodeNumber, lb, ub, LBD, UBD], [], ..]
        # Each node is a list.
        active_set = []

        while not terminate:

            # End it early
            terminate = True

    def objective_callback(self, xI):
        """ Callback for main problem evaluation."""
        obj_surrogate = self.obj_surrogate

        # When run stanalone, the objective is the model objective.
        if self.standalone:
            if self.options['use_surrogate']:
                f = obj_surrogate.predict(xI)

            else:
                system = self.root
                metadata = self.metadata

                # Pass in new parameters
                for var in self.dvs:
                    i, j = self.idx_cache[var]
                    self.set_desvar(var, xI[i:j])

                update_local_meta(metadata, (self.iter_count, ))

                with system._dircontext:
                    system.solve_nonlinear(metadata=metadata)

                # Get the objective function evaluations
                for name, obj in self.get_objectives().items():
                    f = obj
                    break

                self.con_cache = self.get_constraints()

                # Record after getting obj and constraints to assure it has been
                # gathered in MPI.
                self.recorders.record_iteration(system, metadata)

        # When run under AMEIGO, objecitve is the expected improvment
        # function with modifications to make it concave.
        else:
            #ModelInfo_obj=param[0];ModelInfo_g=param[1];con_fac=param[2];flag=param[3]

            X = obj_surrogate.X
            k = np.shape(X)[1]
            lb = self.lb
            ub = self.ub

            # Normalized as per the convention in Kriging of openmdao
            xval = (xI - obj_surrogate.X_mean)/obj_surrogate.X_std

            NegEI = calc_conEI_norm(xval, ModelInfo_obj)

            M=len(ModelInfo_g)
            EV = np.zeros([M,1])
            if M>0:
                # Expected violation evaluation goes here
                for mm in xrange(M):
                    print("Eval con")

            conNegEI = NegEI/(1.0+np.sum(EV))
            P = 0.0

            if self.options['concave_EI']: #Locally makes ei concave to get rid of flat objective space
                for ii in range(k):
                    P += con_fac[ii]*(lb[ii] - xval[ii])*(ub[ii] - xval[ii])

            f = conNegEI + P

        return f