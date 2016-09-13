
import numpy as np
from scipy.sparse import csc_matrix, csc_matrix, coo_matrix, eye as sp_eye

from six.moves import range

def assemble_sparse_jacobian(group, mode='fwd', method='assemble', mult=None,
                             solve_method='solve'):
    """ Assemble and return a sparse array containing the Jacobian for
    the given group.

    Args
    ----
    group : `Group`
        Parent `Group` object.

    mode : string('fwd')
        Derivative mode, can be 'fwd' or 'rev'.

    mult : function(None)
        Solver mult function to coordinate the matrix vector product

    solve_method : string('solve')
        Method of solution of the Ax=b system.  This determines the form
        (CSR or CSC) of the sparse jacobian returned.

    """
    u_vec = group.unknowns
    n_edge = u_vec.vec.size

    # OpenMDAO does matrix vector product.
    if method == 'MVP':

        ident_vec = np.zeros(n_edge)
        icache = None

        partials = np.empty((n_edge, n_edge))

        for i in range(n_edge):
            ident_vec[i-1] = 0.0
            ident_vec[i] = 1.0

            # if the idea here is that partials would be allocated once and
            # from then on would just have its values updated, this column
            # assignment operation could end up being inefficient, and we
            # may be ahead just to create a new partial maxtrix each time
            # using bmat or something.
            partials[:, i] = mult(ident_vec)

    # Assemble the Jacobian
    else:
        rows = []
        cols = []
        data = []

        # keep a set of diagonal entries with values set by sub-jacobians.
        # With dense matrices, identity values on the diagonal would just be overwritten,
        # but if we have entries at duplicate indices when we create our COO matrix,
        # it will add the values together and give us the wrong jacobian.
        diag_exclude = set()

        icache = group._icache
        conn = group.connections
        prom_names = group._sysdata.to_prom_name

        for sub in group.components(recurse=True):

            jac = sub._jacobian_cache

            # This method won't work on components where apply_linear
            # is overridden.
            if jac is None:
                msg = "The 'assemble' jacobian_method is not supported when " + \
                     "'apply_linear' is used on a component (%s)." % sub.pathname
                raise RuntimeError(msg)

            sub_u = sub.unknowns
            sub_name = sub.pathname

            for key in jac:
                o_var, i_var = key
                key2 = (sub_name, key)

                # We cache the location of each variable in our jacobian
                if key2 not in icache:

                    i_var_abs = '.'.join((sub_name, i_var))
                    o_var_abs = '.'.join((sub_name, o_var))
                    i_var_pro = prom_names[i_var_abs]
                    o_var_pro = prom_names[o_var_abs]

                    # non-state inputs need to find their source.
                    if i_var not in sub.states and i_var_pro not in u_vec:

                        # Param is not connected, so it's not relevant
                        if i_var_abs not in conn:
                            continue

                        i_var_src, idxs = conn[i_var_abs]
                        i_var_pro = prom_names[i_var_src]
                    else:
                        idxs = None

                    o_start, o_end = u_vec._dat[o_var_pro].slice
                    i_start, i_end = u_vec._dat[i_var_pro].slice

                    icache[key2] = (o_start, o_end, i_start, i_end, idxs)

                else:
                    (o_start, o_end, i_start, i_end, idxs) = icache[key2]

                # if the current input is only connected to certain entries
                # in its source, we have to map the sub-jacobian to the
                # appropriate rows of the big jacobian.
                if idxs:
                    nidxs = len(idxs)
                    for i in range(o_start, o_end):
                        rows.extend([i]*nidxs)
                        tmp = [i_start+idx for idx in idxs]
                        diag_exclude.update(ii for ii in tmp if ii==i)
                        cols.extend(tmp)
                else:
                    for i in range(o_start, o_end):
                        rows.extend([i]*(i_end-i_start))
                        if i_start <= i and i_end > i:
                            tmp = list(range(i_start, i_end))
                            diag_exclude.update(ii for ii in tmp if ii==i)
                            cols.extend(tmp)
                        else:
                            cols.extend(range(i_start, i_end))
                data.append(jac[key].flatten())

        # set diagonal entries that haven't been overridden to -1
        tmp = [i for i in range(n_edge) if i not in diag_exclude]
        rows.extend(tmp)
        cols.extend(tmp)
        data.append(np.array([-1.0]*len(tmp)))

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.hstack(data)
        partials = coo_matrix((data, (rows, cols)), shape=(n_edge, n_edge))

        if mode == 'fwd' and solve_method == 'LU':
            partials = partials.tocsc()  # CSC needed for efficient LU solve
        else:
            partials = partials.tocsr()

        if mode == 'rev':
            partials = partials.T  # CSR.T -> CSC

    return partials, icache
