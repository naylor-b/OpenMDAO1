""" Non-linear solver that implements a Newton's method."""

from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.util.record_util import update_local_meta, create_local_meta


class Newton(NonLinearSolver):
    """A python Newton solver with line-search adapation of the relaxation
    parameter.
    """

    def __init__(self):
        super(Newton, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-12,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-10,
                       desc='Relative convergence tolerance.')
        opt.add_option('maxiter', 20,
                       desc='Maximum number of iterations.')
        opt.add_option('ls_atol', 1e-10,
                       desc='Absolute convergence tolerancee for line search.')
        opt.add_option('ls_rtol', 0.9,
                       desc='Relative convergence tolerancee for line search.')
        opt.add_option('ls_maxiter', 10,
                       desc='Maximum number of line searches.')
        opt.add_option('alpha', 1.0,
                       desc='Initial over-relaxation factor.')

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Solves the system using a Netwon's Method.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        atol = self.options['atol']
        rtol = self.options['rtol']
        maxiter = self.options['maxiter']
        ls_atol = self.options['ls_atol']
        ls_rtol = self.options['ls_rtol']
        ls_maxiter = self.options['ls_maxiter']
        alpha = self.options['alpha']

        # Metadata setup
        self.iter_count = 0
        ls_itercount = 0
        local_meta = create_local_meta(metadata, system.name)
        system.ln_solver.local_meta = local_meta
        update_local_meta(local_meta, (self.iter_count, ls_itercount))

        # Perform an initial run to propagate srcs to targets.
        system.children_solve_nonlinear(local_meta)
        system.apply_nonlinear(params, unknowns, resids)

        f_norm = resids.norm()
        f_norm0 = f_norm

        if self.options['iprint'] > 0:
            self.print_norm('NEWTON', local_meta, 0, f_norm, f_norm0)

        arg = system.drmat[None]
        result = system.dumat[None]

        alpha_base = alpha
        while self.iter_count < maxiter and f_norm > atol and \
                f_norm/f_norm0 > rtol:

            # Linearize Model
            system.jacobian(params, unknowns, resids)

            # Calculate direction to take step
            arg.vec[:] = resids.vec[:]
            system.solve_linear(system.dumat, system.drmat, [None], mode='fwd')

            unknowns.vec[:] += alpha*result.vec[:]

            # Metadata update
            self.iter_count += 1
            ls_itercount = 0
            update_local_meta(local_meta, (self.iter_count, ls_itercount))

            # Just evaluate the model with the new points
            system.children_solve_nonlinear(local_meta)
            system.apply_nonlinear(params, unknowns, resids, local_meta)

            for recorder in self.recorders:
                recorder.raw_record(params, unknowns, resids, local_meta)

            f_norm = resids.norm()
            if self.options['iprint'] > 0:
                self.print_norm('NEWTON', local_meta, self.iter_count, f_norm, f_norm0)

            # Backtracking Line Search
            while ls_itercount < ls_maxiter and \
                    f_norm > ls_atol and \
                    f_norm/f_norm0 > ls_rtol:

                alpha *= 0.5
                unknowns.vec[:] -= alpha*result.vec[:]
                ls_itercount += 1

                # Metadata update
                update_local_meta(local_meta, (self.iter_count, ls_itercount))

                # Just evaluate the model with the new points

                system.children_solve_nonlinear(local_meta)
                system.apply_nonlinear(params, unknowns, resids, local_meta)

                for recorder in self.recorders:
                    recorder.raw_record(params, unknowns, resids, local_meta)

                f_norm = resids.norm()
                if self.options['iprint'] > 1:
                    self.print_norm('BK_TKG', local_meta, ls_itercount, f_norm,
                                    f_norm/f_norm0, indent=1, solver='LS')

            # Reset backtracking
            alpha = alpha_base

            for recorder in self.recorders:
                recorder.raw_record(params, unknowns, resids, local_meta)

        # Need to make sure the whole workflow is executed at the final
        # point, not just evaluated.
        #self.iter_count += 1
        #update_local_meta(local_meta, (self.iter_count, 0))
        #system.children_solve_nonlinear(local_meta)

        if self.options['iprint'] > 0:
            self.print_norm('NEWTON', local_meta, self.iter_count, f_norm,
                            f_norm0, msg='Converged')

from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel

class GSNewtonHybrid(Newton):

    def __init__(self):
        super(GSNewtonHybrid, self).__init__()
        self.gs_solver = NLGaussSeidel()
        self.gs_solver.options['maxiter'] = 10
        self.gs_solver.options['iprint'] = 2

    def solve(self, params, unknowns, resids, system, metadata=None):
        self.gs_solver.solve(params, unknowns, resids, system, metadata)
        super(GSNewtonHybrid, self).solve(params, unknowns, resids, system, metadata)
