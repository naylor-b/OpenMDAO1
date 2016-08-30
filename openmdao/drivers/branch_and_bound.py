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
from scipy.optimize import minimize
from scipy.special import erf

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
        opt.add_option('active_tol', 1.0e-6, lower=0.0,
                       desc='Tolerance (2-norm) for triggering active set '
                       'reduction.')
        opt.add_option('atol', 1.0e-6, lower=0.0,
                       desc='Absolute tolerance (inf-norm) of upper minus '
                       'lower bound for termination.')
        opt.add_option('concave_EI', False,
                       desc='Set to True to apply a transformation to make the '
                       'objective function concave.')
        opt.add_option('disp', True,
                       desc='Set to False to prevent printing of iteration '
                       'messages.')
        opt.add_option('ftol', 1.0e-12, lower=0.0,
                       desc='Absolute tolerance for sub-optimizations.')
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
        self.con_surrogate = []
        self.record_name = 'B&B'
        self.con_cache = None

        # Set to True if we have found a minimum.
        self.eflag_MINLPBB = False

        self.xopt = None
        self.fopt = None

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
                size = len(np.asarray(val.val))
            except TypeError:
                size = 1
            self.idx_cache[name] = (j, j+size)
            j += size
        self.size = j

        # Lower and Upper bounds
        self.xI_lb = np.empty((self.size))
        self.xI_ub = np.empty((self.size))
        dv_dict = self._desvars
        for var in self.dvs:
            i, j = self.idx_cache[var]
            self.xI_lb[i:j] = dv_dict[var]['lower']
            self.xI_ub[i:j] = dv_dict[var]['upper']

    def run(self, problem):
        """Execute the Branch_and_Bound method.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """
        obj_surrogate = self.obj_surrogate
        atol = self.options['atol']
        active_tol = self.options['active_tol']
        ftol = self.options['ftol']
        integer_tol = self.options['integer_tol']
        disp = self.options['disp']

        # Metadata Setup
        self.metadata = create_local_meta(None, self.record_name)
        self.iter_count = 1
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

                xx_i = np.empty((self.size, ))
                for var in self.dvs:
                    lower = self._desvars[var]['lower']
                    upper = self._desvars[var]['upper']
                    i, j = self.idx_cache[var]
                    x_i_0 = self.sampling[var][i_train, :]

                    xx_i[i:j] = np.round(lower + x_i_0 * (upper - lower))
                    
                x_i.append(xx_i)

            # Run each case and extract obj/con
            cons = {}
            for i_run in range(len(x_i)):

                # Set design variables
                for var in self.dvs:
                    i, j = self.idx_cache[var]
                    self.set_desvar(var, x_i[i_run][i:j])

                with system._dircontext:
                    system.solve_nonlinear(metadata=metadata)

                # Get objectives and constraints
                current_objs = self.get_objectives()
                obj_name = list(current_objs.keys())[0]
                current_obj = current_objs[obj_name].copy()
                obj.append(current_obj)
                current_cons = self.get_constraints()
                cons.append(current_cons)

            self.obj_surrogate = obj_surrogate = self.surrogate()
            obj_surrogate.train(x_i, obj)
            obj_surrogate.y = obj

            for name, val in iteritems(current_cons):
                con_surr = self.con_surrogate.append(self.surrogate())
                con_surr.train(x_i, val)
                con_surr.y = val
                con_surr._name = name

        # Calculate intermediate statistics. This stuff used to be stored in
        # the Modelinfo object, but more convenient to store it in the
        # Kriging surrogate.
        if obj_surrogate:

            #This value should always be 0.0
            obj_surrogate.mu = np.mean(obj_surrogate.Y)

            #This value should always be 1.0
            obj_surrogate.SigmaSqr = obj_surrogate.sigma2/np.square(obj_surrogate.Y_std)

            # TODO: mvp in Kriging
            obj_surrogate.R_inv = obj_surrogate.Vh.T.dot(np.einsum('i,ij->ij',
                                                                   obj_surrogate.S_inv,
                                                                   obj_surrogate.U.T))
            # TODO: what is this?
            obj_surrogate.p = 2

            obj_surrogate.c_r = obj_surrogate.alpha

            # Thsi is also done in Ameigo. TODO: just do it once.
            obj_surrogate.y_best = np.min(obj_surrogate.y)

            #obj_surrogate.X = surrogate.X
            #obj_surrogate.ynorm = surrogate.Y
            #obj_surrogate.thetas = surrogate.thetas
            #obj_surrogate.X_std = obj_surrogate.X_std.reshape(num_xI,1)
            #obj_surrogate.X_mean = obj_surrogate.X_mean.reshape(num_xI,1)

        #----------------------------------------------------------------------
        # Step 1: Initialize
        #----------------------------------------------------------------------

        terminate = False
        num_des = len(self.xI_lb)
        self.eflag_MINLPBB = False
        node_num = 0

        # Initial B&B bounds are infinite.
        LBD = -np.inf
        UBD = np.inf
        LBD_prev =- np.inf

        xL_iter = self.xI_lb.copy()
        xU_iter = self.xI_ub.copy()

        # Optimal objective and solution
        fopt = np.inf
        xopt = np.empty(xL_iter.shape)

        # This stuff is just for printing.
        par_node = 0

        # Active set fields:
        #     Aset = [[NodeNumber, lb, ub, LBD, UBD], [], ..]
        # Each node is a list.
        active_set = []

        while not terminate:
            int_flag = False

            # Reduce the active set if the upper and lower bounds are really
            # close. This is not considered an independent iteration.
            if np.linalg.norm(xL_iter - xU_iter) <= active_tol:
                xloc_iter = xL_iter.copy()
                floc_iter = self.objective_callback(xloc_iter)

                #Better solution found
                if floc_iter < UBD:
                    UBD = floc_iter
                    fopt = UBD
                    xopt = xloc_iter.copy().reshape(num_des)
                    self.eflag_MINLPBB = True

                    # Update active set: Removes the current node
                    if len(active_set) >= 1:
                        active_set = update_active_set(active_set, UBD)

            # Branch and Bound main iteration.
            else:

                #--------------------------------------------------------------
                #Step 2: Obtain a local solution
                #--------------------------------------------------------------

                # Using a gradient-based method here.
                # TODO: Make it more pluggable.
                xC_iter = (0.5*(xU_iter + xL_iter)).reshape((num_des, 1))
                bnds = [(xL_iter[ii], xU_iter[ii]) for ii in xrange(num_des)]

                optResult = minimize(self.objective_callback, xC_iter,
                                     method='SLSQP', bounds=bnds,
                                     options={'ftol' : ftol})

                xloc_iter = optResult.x.reshape(num_des, 1)
                floc_iter = optResult.fun

                if not optResult.success:
                    efloc_iter = 0
                    floc_iter = np.inf
                else:
                    efloc_iter = 1

                # Round any close to integer value to integer solution.
                for aa in range(len(xloc_iter)):
                    rounded = np.round(xloc_iter[aa])
                    if np.abs(rounded - xloc_iter[aa]) <= integer_tol:
                        xloc_iter[aa] = rounded

                # Reduce the active set if an Integer solution was found.
                if np.linalg.norm(xloc_iter - np.round(xloc_iter)) <= active_tol and \
                   efloc_iter >= 1:
                    int_flag = True

                    # Better integer solution found
                    if floc_iter < UBD:
                        UBD = 1.0*floc_iter
                        fopt = 1.0*UBD
                        xopt = xloc_iter.copy().reshape(num_des)
                        self.eflag_MINLPBB = True

                        # Update active set
                        if len(active_set) >= 1:
                            active_set = update_active_set(active_set, UBD)

                #--------------------------------------------------------------
                # Step 3: Partition the current rectangle as per the new
                # branching scheme.
                #--------------------------------------------------------------

                child_info = np.zeros([2,3])
                dis_flag = [' ',' ']
                for ii in range(2):
                    lb = xL_iter.copy()
                    ub = xU_iter.copy()

                    if efloc_iter >= 1:
                        #Case 1: Not an integer sol. Branch at variable farthest from the integer value
                        if int_flag is False:
                            l_iter = np.abs(np.round(xloc_iter) - xloc_iter).argmax()
                            if ii == 0:
                                ub[l_iter] = np.floor(xloc_iter[l_iter])
                            elif ii == 1:
                                lb[l_iter] = np.ceil(xloc_iter[l_iter])

                        #Case 2: Integer sol. Branch at variable with the largest edge
                        else:
                            l_iter = (xU_iter - xL_iter).argmax()
                            if ii == 0:
                                if lb[l_iter] < xloc_iter[l_iter]:
                                    #Bound shifted by 1 unit to the left
                                    ub[l_iter] = xloc_iter[l_iter] - 1
                                else:
                                    ub[l_iter] = 1.0*lb[l_iter]
                            elif ii == 1:
                                if ub[l_iter] > xloc_iter[l_iter]:
                                    #Bound shifted by 1 unit to the right
                                    lb[l_iter] = xloc_iter[l_iter] + 1
                                else:
                                    lb[l_iter] = 1.0*ub[l_iter]
                    else:
                        #Branch at variable with largest edge
                        l_iter = (xU_iter - xL_iter).argmax()
                        if ii == 0:
                            ub[l_iter] = np.floor(0.5*(xU_iter[l_iter] + xL_iter[l_iter]))
                        elif ii == 1:
                            lb[l_iter] = np.ceil(0.5*(xU_iter[l_iter] + xL_iter[l_iter]))

                    #--------------------------------------------------------------
                    # Step 4: Obtain an LBD of f in the newly created node
                    #--------------------------------------------------------------

                    S4_fail = False
                    x_comL, x_comU, Ain_hat, bin_hat = gen_coeff_bound(lb, ub, obj_surrogate)
                    sU, eflag_sU = self.maximize_S(x_comL, x_comU, Ain_hat, bin_hat,
                                                   obj_surrogate)

                    if eflag_sU:
                        yL, eflag_yL = self.minimize_y(x_comL, x_comU, Ain_hat, bin_hat,
                                                       obj_surrogate)

                        if eflag_yL:
                            NegEI = calc_conEI_norm([], obj_surrogate, SSqr=sU, y_hat=yL)

                            M = len(self.con_surrogate)
                            EV = np.zeros([M, 1])
                            if M>0:
                                # Expected violation goes here
                                for mm in range(M):
                                    x_comL, x_comU, Ain_hat, bin_hat = gen_coeff_bound(lb, ub, con_surrogate[mm])
                                    sU_g, eflag_sU_g = maximize_S(x_comL, x_comU, Ain_hat,
                                                                  bin_hat, con_surrogate[mm])
                                    sL_g = -1.*sU_g
                                    if eflag_sU_g >= 1:
                                        yL_g, eflag_yL_g = minimize_y(x_comL, x_comU, Ain_hat,
                                                                      bin_hat, con_surrogate[mm])
                                        if eflag_yL_g >= 1:
                                            EV[mm] = calc_conEV_norm([], con_surrogate[mm],
                                                                     gSSqr=sL_g, g_hat=yL_g)
                                        else:
                                            S4_fail = 1.
                                            break
                                    else:
                                        S4_fail = 1
                        else:
                            if disp:
                                print("Cannot solve Min y_hat problem!")
                            S4_fail = True
                    else:
                        if disp:
                            print("Cannot solve Max S problem!")
                        S4_fail = True

                    # Convex approximation failed!
                    if S4_fail:
                        if efloc_iter >= 1:
                            LBD_NegConEI = LBD_prev
                        else:
                            LBD_NegConEI = np.inf
                        dis_flag[ii] = 'F'
                    else:
                        LBD_NegConEI = (NegEI/(1.0 + np.sum(EV)))

                    #--------------------------------------------------------------
                    # Step 5: Store any new node inside the active set that has LBD
                    # lower than the UBD.
                    #--------------------------------------------------------------

                    if (LBD_NegConEI) < UBD:
                        node_num += 1
                        new_node = [node_num, lb, ub, LBD_NegConEI, floc_iter]
                        active_set.extend([new_node])
                        child_info[ii] = np.array([node_num, LBD_NegConEI, floc_iter])
                    else:
                        child_info[ii] = np.array([par_node, LBD_NegConEI, floc_iter])

                if disp:
                    if (self.iter_count-1) % 25 == 0:
                        # Display output in a tabular format
                        print("="*85)
                        print("%19s%12s%14s%21s" % ("Global", "Parent", "Child1", "Child2"))
                        template = "%s%8s%10s%8s%9s%11s%10s%11s%11s"
                        print(template % ("Iter", "LBD", "UBD", "Node", "Node1", "LBD1",
                                          "Node2", "LBD2", "Flocal"))
                        print("="*85)
                    template = "%3d%10.2f%10.2f%6d%8d%1s%13.2f%8d%1s%13.2f%9.2f"
                    print(template % (self.iter_count, LBD, UBD, par_node, child_info[0, 0],
                                      dis_flag[0], child_info[0, 1], child_info[1, 0],
                                      dis_flag[1], child_info[1, 1], child_info[1, 2]))

            # Termination
            if len(active_set) >= 1:
                # Update LBD and select the current rectangle

                # a. Set LBD as lowest in the active set
                LBD = min([active_set[ii][3] for ii in range(len(active_set))])
                ind_LBD = [active_set[ii][3] for ii in range(len(active_set))].index(LBD)
                LBD_prev = LBD

                # b. Select the lowest LBD node as the current node
                xL_iter = active_set[ind_LBD][1]
                xU_iter = active_set[ind_LBD][2]
                par_node = active_set[ind_LBD][0]
                self.iter_count += 1

                # c. Delete the selected node from the Active set of nodes
                del active_set[ind_LBD]

                #--------------------------------------------------------------
                #Step 7: Check for convergence
                #--------------------------------------------------------------
                diff = np.abs(UBD - LBD)
                if diff < atol:
                    terminate = True
                    if disp:
                        print("="*85)
                        print("Terminating! Absolute difference between the upper " + \
                              "and lower bound is below the tolerence limit.")
            else:
                terminate = True
                if disp:
                    print("="*85)
                    print("Terminating! No new node to explore.")

        # Finalize by putting optimal value back into openMDAO
        if self.standalone:

            for var in self.dvs:
                i, j = self.idx_cache[var]
                self.set_desvar(var, xopt[i:j])

            update_local_meta(metadata, (self.iter_count, ))

            with system._dircontext:
                system.solve_nonlinear(metadata=metadata)

        else:
            self.xopt = xopt
            self.fopt = fopt

    def objective_callback(self, xI):
        """ Callback for main problem evaluation."""
        obj_surrogate = self.obj_surrogate

        # When run stanalone, the objective is the model objective.
        if self.standalone:
            if self.options['use_surrogate']:
                f = obj_surrogate.predict(xI)[0]

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
            lb = self.xI_lb
            ub = self.xI_ub

            # Normalized as per the convention in Kriging of openmdao
            xval = (xI - obj_surrogate.X_mean)/obj_surrogate.X_std

            NegEI = calc_conEI_norm(xval, obj_surrogate)

            con_surrogate = self.con_surrogate
            M = len(con_surrogate)
            EV = np.zeros([M, 1])
            if M>0:
                for mm in range(M):
                    EV[mm] = calc_conEV_norm(xval, con_surrogate[mm])

            conNegEI = NegEI/(1.0+np.sum(EV))
            P = 0.0

            if self.options['concave_EI']: #Locally makes ei concave to get rid of flat objective space
                for ii in range(k):
                    P += con_fac[ii]*(lb[ii] - xval[ii])*(ub[ii] - xval[ii])

            f = conNegEI + P

        #print(xI, f)
        return f

    def maximize_S(self, x_comL, x_comU, Ain_hat, bin_hat, surogate):
        """This method finds an upper bound to the SigmaSqr Error, and scales
        up 'r' to provide a smooth design space for gradient-based approach.
        """
        R_inv = surogate.R_inv
        SigmaSqr = surogate.SigmaSqr
        X = surogate.X

        n, k = X.shape
        one = np.ones([n, 1])

        xhat_comL = x_comL.copy()
        xhat_comU = x_comU.copy()
        xhat_comL[k:] = np.zeros([n, 1])
        xhat_comU[k:] = np.ones([n, 1])

        # Calculate the convexity factor alpha
        rL = x_comL[k:]
        rU = x_comU[k:]

        dr_drhat = np.zeros([n, n])
        for ii in range(n):
            dr_drhat[ii, ii] = rU[ii, 0] - rL[ii, 0]

        T2_num = np.dot(np.dot(R_inv, one),np.dot(R_inv, one).T)
        T2_den = np.dot(one.T, np.dot(R_inv, one))
        d2S_dr2 = 2.0*SigmaSqr*(R_inv - (T2_num/T2_den))
        H_hat = np.dot(np.dot(dr_drhat, d2S_dr2), dr_drhat.T)

        # Use Gershgorin's circle theorem to find a lower bound of the
        # min eigen value of the hessian
        eig_lb = np.zeros([n, 1])
        for ii in xrange(n):
            dia_ele = H_hat[ii, ii]
            sum_rw = 0.0
            sum_col = 0.0
            for jj in range(n):
                if ii != jj:
                    sum_rw += np.abs(H_hat[ii,jj])
                    sum_col += np.abs(H_hat[jj,ii])

                eig_lb[ii] = dia_ele - np.min(np.array([sum_rw, sum_col]))

        eig_min = np.min(eig_lb)
        alpha = np.max(np.array([0.0, -0.5*eig_min]))

        # Just storing it here to pull it out in the callback?
        surogate._alpha = alpha

        # Maximize S
        x0 = 0.5*(xhat_comL + xhat_comU)
        bnds = [(xhat_comL[ii], xhat_comU[ii]) for ii in xrange(len(xhat_comL))]

        #Note: Python defines constraints like g(x) >= 0
        cons = [{'type' : 'ineq',
                 'fun' : lambda x : -np.dot(Ain_hat[ii, :],x) + bin_hat[ii],
                 'jac' : lambda x : -Ain_hat[ii, :]} for ii in xrange(2*n)]

        optResult = minimize(self.calc_SSqr_convex, x0,
                             args=(x_comL,x_comU,xhat_comL,xhat_comU),
                             method='SLSQP', constraints=cons, bounds=bnds,
                             options={'ftol' : self.options['ftol'],
                                      'maxiter' : 100})

        Neg_sU = optResult.fun
        if not optResult.success:
            eflag_sU = False
        else:
            eflag_sU = True
            #for ii in range(2*n):
                #if np.dot(Ain_hat[ii, :], optResult.x) > (bin_hat[ii ,0] + 1.0e-6):
                    #eflag_sU = False
                    #break

        sU = - Neg_sU
        return sU, eflag_sU

    def calc_SSqr_convex(self, x_com, *param):
        obj_surrogate = self.obj_surrogate
        x_comL = param[0]
        x_comU = param[1]
        xhat_comL = param[2]
        xhat_comU = param[3]

        X = obj_surrogate.X
        R_inv = obj_surrogate.R_inv
        SigmaSqr = obj_surrogate.SigmaSqr
        alpha = obj_surrogate._alpha

        n, k = X.shape
        one = np.ones([n, 1])
        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = np.array([x_com[k:]]).reshape(n, 1)

        r = rL + rhat*(rU - rL)
        rhat_L = xhat_comL[k:]
        rhat_U = xhat_comU[k:]
        term1 = -SigmaSqr*(1.0 - r.T.dot(np.dot(R_inv, r)) + \
        ((1.0-one.T.dot(np.dot(R_inv, r)))**2/(one.T.dot(np.dot(R_inv, one)))))

        term2 = alpha*np.sum((rhat-rhat_L)*(rhat-rhat_U))
        S2 = term1 + term2
        return S2[0,0]

    def minimize_y(self, x_comL, x_comU, Ain_hat, bin_hat, surrogate):

        # 1- Formulates y_hat as LP (weaker bound)
        # 2- Uses non-convex relaxation technique (stronger bound) [Future release]
        app = 1

        X = surrogate.X
        n = np.shape(X)[0]
        k = np.shape(X)[1]

        xhat_comL = x_comL.copy()
        xhat_comU = x_comU.copy()
        xhat_comL[k:] = np.zeros([n, 1])
        xhat_comU[k:] = np.ones([n, 1])

        if app == 1:
            x0 = 0.5*(xhat_comL + xhat_comU)
            bnds = [(xhat_comL[ii], xhat_comU[ii]) for ii in xrange(len(xhat_comL))]

            cons = [{'type' : 'ineq',
                     'fun' : lambda x : -np.dot(Ain_hat[ii, :],x) + bin_hat[ii],
                     'jac': lambda x: -Ain_hat[ii, :]} for ii in xrange(2*n)]

            optResult = minimize(self.calc_y_hat_convex, x0,
                                 args=(x_comL, x_comU), method='SLSQP',
                                 constraints=cons, bounds=bnds,
                                 options={'ftol' : self.options['ftol'],
                                          'maxiter' : 100})
            yL = optResult.fun

            if not optResult.success:
                eflag_yL = False
            else:
                eflag_yL = True
                #for ii in range(2*n):
                    #if np.dot(Ain_hat[ii, :], optResult.x) > (bin_hat[ii, 0] + 1.0e-6):
                        #eflag_yL = False
                        #break

        return yL, eflag_yL


    def calc_y_hat_convex(self, x_com, *param):
        obj_surrogate = self.obj_surrogate
        x_comL = param[0]
        x_comU = param[1]

        X = obj_surrogate.X
        c_r = obj_surrogate.c_r
        mu = obj_surrogate.mu
        n, k = X.shape

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = np.array([x_com[k:]]).reshape(n, 1)
        r = rL + rhat*(rU - rL)

        y_hat = mu + np.dot(r.T, c_r)
        return y_hat[0, 0]


def update_active_set(active_set, ubd):
    """ Remove variables from the active set data structure if their current
    upper bound exceeds the given value.

    Args
    ----
    active_set : list of lists of floats
        Active set data structure of form [[NodeNumber, lb, ub, LBD, UBD], [], ..]
    ubd : float
        Maximum for bounds test.

    Returns
    -------
    new active_set
    """
    del_flag = []
    for aaa in range(len(active_set)):
        if active_set[aaa][3] >= ubd:
            del_flag.extend([aaa])
    return [ii for jj, ii in enumerate(active_set) if jj not in del_flag]


def gen_coeff_bound(xI_lb, xI_ub, surrogate):
    """This function generates the upper and lower bound of the artificial
    variable r and the coefficients for the linearized under estimator
    constraints. The version accepts design bound in the original design
    space, converts it to normalized design space.
    """

    #Normalized to 0-1 hypercube
    # xL_hat0 = (xI_lb - surrogate.lb_org)/(surrogate.ub_org - surrogate.lb_org)
    # xU_hat0 = (xI_ub - surrogate.lb_org)/(surrogate.ub_org - surrogate.lb_org)

    #Normalized as per Openmdao kriging model
    xL_hat = (xI_lb - surrogate.X_mean)/surrogate.X_std
    xU_hat = (xI_ub - surrogate.X_mean)/surrogate.X_std

    rL, rU = interval_analysis(xL_hat, xU_hat, surrogate)

    # Combined design variables for supbproblem
    num = len(xL_hat) + len(rL)
    x_comL = np.append(xL_hat,rL).reshape(num, 1)
    x_comU = np.append(xU_hat,rU).reshape(num, 1)

    # Coefficients of the linearized constraints of the subproblem
    Ain_hat, bin_hat = lin_underestimator(x_comL, x_comU, surrogate)

    return x_comL, x_comU, Ain_hat, bin_hat


def interval_analysis(lb_x, ub_x, surrogate):
    """ The module predicts the lower and upper bound of the artificial
    variable 'r' from the bounds of the design variable x r is related to x
    by the following equation:

    r_i = exp(-sum(theta_h*(x_h - x_h_i)^2))

    """

    X = surrogate.X
    thetas = surrogate.thetas
    p = surrogate.p
    n, k = X.shape

    t1L = np.zeros([n, k]); t1U = np.zeros([n, k])
    t2L = np.zeros([n, k]); t2U = np.zeros([n, k])
    t3L = np.zeros([n, k]); t3U = np.zeros([n, k])
    t4L = np.zeros([n, 1]); t4U = np.zeros([n, 1])
    lb_r = np.zeros([n, 1]); ub_r = np.zeros([n, 1])

    eterm = 1
    if p % 2 == 0:
        for i in xrange(n):
            for h in xrange(k):
                t1L[i,h] = lb_x[h] - X[i,h]
                t1U[i,h] = ub_x[h] - X[i,h]

                t2L[i,h] = np.max(np.array([0,np.min(np.array([t1L[i,h]*t1L[i,h],
                                                                t1L[i,h]*t1U[i,h],
                                                                t1U[i,h]*t1U[i,h]]))]))
                t2U[i,h] = np.max(np.array([0,np.max(np.array([t1L[i,h]*t1L[i,h],
                                                                t1L[i,h]*t1U[i,h],
                                                                t1U[i,h]*t1U[i,h]]))]))

                t3L[i,h] = np.min(np.array([-thetas[h]*t2L[i,h], -thetas[h]*t2U[i,h]]))
                t3U[i,h] = np.max(np.array([-thetas[h]*t2L[i,h], -thetas[h]*t2U[i,h]]))

            t4L[i] = np.sum(t3L[i,:])
            t4U[i] = np.sum(t3U[i,:])

            lb_r[i] = np.exp(t4L[i])
            ub_r[i] = np.exp(t4U[i])
    else:
        print("\nWarning! Value of p should be 2. Cannot perform interval analysis")
        print("\nReturing global bound of the r variable")
        eterm =0

    return lb_r, ub_r


def lin_underestimator(lb, ub, surrogate):
    X = surrogate.X
    thetas = surrogate.thetas
    p = surrogate.p
    n, k = X.shape

    lb_x = lb[:k]; ub_x = ub[:k]
    lb_r = lb[k:]; ub_r = ub[k:]

    a1 = np.zeros([n, n]); a3 = np.zeros([n, n])
    a1_hat = np.zeros([n, n]); a3_hat = np.zeros([n, n])
    a2 = np.zeros([n, k]); a4 = np.zeros([n, k])
    b2 = np.zeros([n, k]); b4 = np.zeros([n, k])
    b1 = np.zeros([n, 1]); b3 = np.zeros([n, 1])
    b1_hat = np.zeros([n, 1]); b3_hat = np.zeros([n, 1])

    for i in xrange(n):
        #T1: Linearize under-estimator of ln[r_i] = a1[i,i]*r[i] + b1[i]
        if ub_r[i] < 1.0e-323 or (ub_r[i] - lb_r[i]) < 1.0e-308:
            # a1[i,i] = 0.
            # b1[i] = -np.inf
            a1_hat[i,i] = 0.0 #a1[i,i]*(ub_r[i]-lb_r[i])
            b1_hat[i] = -np.inf #a1[i,i]*lb_r[i] + b1[i]
        elif ub_r[i] <= lb_r[i]:
            # a1[i,i] = 0.0
            # b1[i] = np.log(ub_r[i])
            a1_hat[i,i] = 0.0 #a1[i,i]*(ub_r[i]-lb_r[i])
            b1_hat[i] = np.log(ub_r[i]) #a1[i,i]*lb_r[i] + b1[i]
        elif lb_r[i] < 1.0e-323:
            # a1[i,i] = np.inf
            # b1[i] = -np.inf
            a1_hat[i,i] = np.inf #a1[i,i]*(ub_r[i]-lb_r[i])
            b1_hat[i] = -np.inf #b1[i]
        else:
            a1[i,i] = ((np.log(ub_r[i]) - np.log(lb_r[i]))/(ub_r[i] - lb_r[i]))
            b1[i] = np.log(ub_r[i]) - a1[i,i]*ub_r[i]
            a1_hat[i,i] = a1[i,i]*(ub_r[i]-lb_r[i])
            b1_hat[i] = a1[i,i]*lb_r[i] + b1[i]

        #T3: Linearize under-estimator of -ln[r_i] = a3[i,i]*r[i] + b3[i]
        if ub_r[i] < 1.0e-323:
            a3_hat[i,i] = 0.0
            b3_hat[i] = np.inf
        else:
            r_m_i = (lb_r[i] + ub_r[i])/2.0
            if r_m_i < 1e-308:
                a3_hat[i,i] = -np.inf
                b3_hat[i] = np.inf
            else:
                a3[i,i] = -1.0/r_m_i
                b3[i] = -np.log(r_m_i) - a3[i,i]*r_m_i
                a3_hat[i,i] = a3[i,i]*(ub_r[i] - lb_r[i])
                b3_hat[i] = a3[i,i]*lb_r[i] + b3[i]

        for h in xrange(k):
            #T2: Linearize under-estimator of thetas_h*(x_h - X_h_i)^2 = a4[i,h]*x_h[h] + b4[i,h]
            x_m_h = (ub_x[h] + lb_x[h])/2.0
            a2[i,h] = p*thetas[h]*(x_m_h - X[i,h])**(p-1.0)
            yy = thetas[h]*(x_m_h - X[i,h])**p
            b2[i,h] = -a2[i,h]*x_m_h + yy

            #T4: Linearize under-estimator of -theta_h*(x_h - X_h_i)^2 = a4[i,h]*x_h[h] + b4[i,h]
            yy2 = -thetas[h]*(ub_x[h] - X[i,h])**p
            yy1 = -thetas[h]*(lb_x[h] - X[i,h])**p

            if ub_x[h] <= lb_x[h]:
                a4[i,h] = 0.0
            else:
                a4[i,h] = (yy2 - yy1)/(ub_x[h] - lb_x[h])

            b4[i,h] = -a4[i,h]*lb_x[h] + yy1

    Ain1 = np.concatenate((a2, a4), axis=0)
    Ain2 = np.concatenate((a1_hat, a3_hat), axis=0)
    Ain_hat = np.concatenate((Ain1, Ain2), axis=1)
    bin_hat = np.concatenate((-(b1_hat + np.sum(b2, axis=1).reshape(n,1)),
                              -(b3_hat + np.sum(b4, axis=1).reshape(n,1))), axis=0)

    return Ain_hat, bin_hat

def calc_conEI_norm(xval, obj_surrogate, SSqr=None, y_hat=None):
    """This function evaluates the expected improvement in the normalized
    design space.
    """
    y_min = (obj_surrogate.y_best - obj_surrogate.Y_mean)/obj_surrogate.Y_std
    X = obj_surrogate.X

    if not SSqr:
        c_r = obj_surrogate.c_r
        thetas = obj_surrogate.thetas
        SigmaSqr = obj_surrogate.SigmaSqr
        R_inv = obj_surrogate.R_inv
        mu = obj_surrogate.mu
        p = obj_surrogate.p

        n = np.shape(X)[0]
        one = np.ones([n, 1])
        r = np.ones([n, 1])
        for ii in xrange(n):
            r[ii] = np.exp(-np.sum(thetas*(xval - X[ii])**p))

        y_hat = mu + np.dot(r.T,c_r)
        SSqr = SigmaSqr*(1.0 - r.T.dot(np.dot(R_inv, r)) + \
        ((1.0 - one.T.dot(np.dot(R_inv, r)))**2)/(one.T.dot(np.dot(R_inv, one))))

    if SSqr <= 0.0:
        NegEI = 0.0
    else:
        ei1 = (y_min-y_hat)*(0.5+0.5*erf((1/np.sqrt(2))*((y_min-y_hat)/np.sqrt(abs(SSqr)))))
        ei2 = np.sqrt(np.abs(SSqr))*(1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*((y_min-y_hat)**2/np.abs(SSqr)))
        NegEI = -(ei1 + ei2)

    return NegEI


def calc_conEV_norm(xval, con_surrogate, gSSqr=None, g_hat=None):
    """This modules evaluates the expected improvement in the normalized
    design sapce"""

    g_min = 0.0
    X = con_surrogate.X

    if not gSSqr:
        c_r = con_surrogate.c_r
        thetas = con_surrogate.thetas
        SigmaSqr = con_surrogate.SigmaSqr
        R_inv = con_surrogate.R_inv
        mu = con_surrogate.mu
        p = con_surrogate.p
        n = np.shape(X)[0]
        one = np.ones([n,1])
        r = np.ones([n,1])
        for ii in xrange(n):
            r[ii] = np.exp(-np.sum(thetas*(xval - X[ii])**p))

        g_hat = mu + np.dot(r.T,c_r)
        gSSqr = SigmaSqr*(1.0 - r.T.dot(np.dot(R_inv, r)) + \
        ((1.0 - one.T.dot(np.dot(R_inv, r)))**2)/(one.T.dot(np.dot(R_inv, one))))

    if gSSqr <= 0:
        EV = 0.0
    else:
        # Calculate expected violation
        ei1 = (g_hat-g_min)*(0.5+0.5*erf((1.0/np.sqrt(2.0))*((g_hat-g_min)/np.sqrt(np.abs(gSSqr)))))
        ei2 = np.sqrt(np.abs(gSSqr))*(1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*((g_hat-g_min)**2/np.abs(gSSqr)))
        EV = (ei1 + ei2)

    return EV