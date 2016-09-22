""" OpenMDAO LinearSolver that explicitly solves the linear system using
linalg.solve or scipy LU factor/solve. Inherits from MultLinearSolver just
for the mult function."""

from collections import OrderedDict

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve, splu
from openmdao.solvers.solver_base import MultLinearSolver
from openmdao.core.jacobian import SparseJacobian, DenseJacobian, MVPJacobian

from six import iteritems

class DirectSolver(MultLinearSolver):
    """ OpenMDAO LinearSolver that explicitly solves the linear system using
    linalg.solve. The user can choose to have the jacobian assembled
    directly or through matrix-vector product.

    Options
    -------
    options['iprint'] :  int(0)
        Set to 0 to print only failures, set to 1 to print iteration totals to
        stdout, set to 2 to print the residual each iteration to stdout,
        or -1 to suppress all printing.
    options['mode'] :  str('auto')
        Derivative calculation mode, set to 'fwd' for forward mode, 'rev' for
        reverse mode, or 'auto' to let OpenMDAO determine the best mode.
    options['jacobian_method'] : str('MVP')
        Method to assemble the jacobian to solve. Select 'MVP' to build the
        Jacobian by calling apply_linear with columns of identity. Select
        'assemble' to build the Jacobian by taking the calculated Jacobians in
        each component and placing them directly into a clean identity matrix.
    options['solve_method'] : str('LU')
        Solution method, either 'solve' for linalg.solve, or 'LU' for
        linalg.lu_factor and linalg.lu_solve.
    options['jacobian_format'] : str('dense')
        Specifies whether the Jacobian will be dense or sparse.
    """

    def __init__(self):
        super(DirectSolver, self).__init__()
        self.options.remove_option("err_on_maxiter")
        self.options.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " +
                       "forward mode, 'rev' for reverse mode, or 'auto' to " +
                       "let OpenMDAO determine the best mode.",
                       lock_on_setup=True)

        self.options.add_option('jacobian_method', 'MVP', values=['MVP', 'assemble'],
                                desc="Method to assemble the jacobian to solve. " +
                                "Select 'MVP' to build the Jacobian by calling " +
                                "apply_linear with columns of identity. Select " +
                                "'assemble' to build the Jacobian by taking the " +
                                "calculated Jacobians in each component and placing " +
                                "them directly into a clean identity matrix.")
        self.options.add_option('solve_method', 'LU', values=['LU', 'solve'],
                                desc="Solution method, either 'solve' for linalg.solve, " +
                                "or 'LU' for linalg.lu_factor and linalg.lu_solve.")
        self.options.add_option('jacobian_format', 'dense', values=['dense', 'sparse'],
                                desc='Specifies either a sparse or dense Jacobian.')

        self.jacobian = None
        self.lup = None
        self.mode = None

    def solve(self, rhs_mat, group, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        Args
        ----
        rhs_mat : dict of ndarray
            Dictionary containing one ndarry per top level quantity of
            interest. Each array contains the right-hand side for the linear
            solve.

        group : `Group`
            Parent `Group` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        dict of ndarray : Solution vectors
        """

        self.system = group

        if self.mode is None:
            self.mode = mode

        sol_buf = OrderedDict()

        self.voi = None

        if group._jacobian_changed or mode != self.mode:
            self.mode = mode
            method = self.options['jacobian_method']

            if method == 'MVP':
                self.jacobian = MVPJacobian(group.unknowns.vec.size, self.mult)
            elif self.options['jacobian_format'] == 'sparse':
                self.jacobian = SparseJacobian(group.unknowns.slice_iter(),
                                               group._sub_jac_iter(), mode)
            else:
                self.jacobian = DenseJacobian(group.unknowns.slice_iter(),
                                              group._sub_jac_iter(), mode)

            group._jacobian_changed = False

            if self.options['solve_method'] == 'LU':
                if issparse(self.jacobian.partials):
                    self.lup = splu(self.jacobian.partials.tocsc())
                else:
                    self.lup = lu_factor(self.jacobian.partials)

        for voi, rhs in rhs_mat.items():

            if self.options['solve_method'] == 'LU':
                if issparse(self.jacobian.partials):
                    deriv = self.lup.solve(rhs)
                else:
                    deriv = lu_solve(self.lup, rhs)
            else:
                if issparse(self.jacobian.partials):
                    deriv = spsolve(self.jacobian.partials, rhs)
                else:
                    deriv = np.linalg.solve(self.jacobian.partials, rhs)
            self.system = None
            sol_buf[voi] = deriv

        return sol_buf
