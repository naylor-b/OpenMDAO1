
import numpy as np
from scipy.sparse import csc_matrix, csc_matrix, coo_matrix, eye as sp_eye
from itertools import product

from six.moves import range

# REQs
# - pre-allocation during setup
# - components will be passed this object and can set sub jacobians into it
#    using  J['a', 'b'] style syntax
# - for dense jacobian w/dense sub-jacobian, component can operate directly
#   on subview of the full jacbian, so no data copying
# - for sparse J w/sparse sub-J, allow (if possible) component data to be
#   a sub-view of sparse matrix data array
# - for mismatches (sparse/dense or dense/sparse) we'll just have to do
#   copying of data


class Jacobian(object):
    def __init__(self, unknowns):
        # record block order and slice for each vector variable in unknowns
        self._ordering = {
            n : (i, acc.slice) for i, (n,acc) in enumerate(iteritems(unknowns))
                   if not acc.pbo
        }

    def _sub_jac_iter(self, group, connections, prom_map):
        """
        A generator of tuples of the form (ovar, ivar, jac, idxs) that
        will iterate over all sub-jacobian entries for components contained
        in the given group.
        """

        for sub in group.components(recurse=True):

            jac = sub._jacobian_cache

            # This method won't work on components where apply_linear
            # is overridden.
            if jac is None:
                msg = "The 'assemble' jacobian_method is not supported when " + \
                     "'apply_linear' is used on a component (%s)." % sub.pathname
                raise RuntimeError(msg)

            sub_u = sub.unknowns
            sub_path = sub.pathname

            for key in jac:
                o_var, i_var = key

                i_var_abs = '.'.join((sub_path, i_var))
                i_var_prom = prom_map[i_var_abs]

                # non-state inputs need to find their source.
                if i_var not in sub.states and i_var_prom not in uvec:

                    # Param is not connected, so it's not relevant
                    if i_var_abs not in connections:
                        continue

                    i_var_src, idxs = connections[i_var_abs]
                    i_var_prom = prom_map[i_var_src]
                else:
                    idxs = None

                o_var_abs = '.'.join((sub_path, o_var))
                o_var_prom = prom_map[o_var_abs]

                yield o_var_prom, i_var_prom, jac[key], idxs

class DenseJacobian(Jacobian):
    def assemble(self):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass


class SparseJacobian(Jacobian):
    def assemble(self, group, connections, prom_map,
                 mode='fwd', solve_method='solve'):
        """ Assemble and return a sparse array containing the Jacobian for
        the given group.

        Args
        ----
        uvec : VecWrapper
            Unknowns vecwrapper.

        components : iter of components
            An iterator of components to loop over and retrieve their
            sub-jacobians.

        connections : dict
            A mapping of abs target name to a tuple of (abs src, idxs)

        prom_map : dict
            A mapping of local name to local promoted name.

        mode : string('fwd')
            Derivative mode, can be 'fwd' or 'rev'.

        solve_method : string('solve')
            Method of solution of the Ax=b system.  This determines the form
            (CSR or CSC) of the sparse jacobian.

        """

        n_edge = uvec.vec.size

        rows = []
        cols = []
        data = []
        data_idxs = []

        data_idx = 0
        data_size = 0

        subJinfo = []

        for ovar, ivar, subjac, idxs in self._sub_jac_iter(group, connections, prom_map):
            data_size += subjac.size

            irow, (o_start, o_end) = self._ordering[ovar]
            icol, (i_start, i_end) = self._ordering[ivar]

            sparse = issparse(subjac)
            if sparse:
                # make sure it's in COO format. This does nothing if it's
                # already COO
                subjac = subjac.tocoo()

            rows = np.empty(subjac.size, dtype=int)
            cols = np.empty(subjac.size, dtype=int)

            # if the current input is only connected to certain entries
            # in its source, we have to map the sub-jacobian to the
            # appropriate rows of the big jacobian.
            if idxs:
                if sparse:
                    cols[:] = subjac.col
                    cols += i_start

                    assert len(idxs) == subjac.shape[0]

                    for i, row in enumerate(subjac.row):
                        # note that if idxs are not in sorted order then row idxs
                        # won't be in sorted order
                        rows[i] = row + o_start + idxs[i]
                else:  # not sparse
                    colrange = np.arange(i_start, i_end, dtype=int)
                    rowrange = np.array(idxs, dtype=int) + o_start
                    ncols = colrange.size
                    nrows = rowrange.size

                    for i, row in enumerate(rowrange):
                        rows[i*ncols:(i+1)*ncols] = np.full(ncols, row, dtype=int)
                        cols[i*ncols:(i+1)*ncols] = colrange
            else:
                if sparse:
                    rows[:] = subjac.row
                    rows += o_start
                    cols[:] = subjac.col
                    cols += i_start
                else:
                    colrange = np.arange(i_start, i_end, dtype=int)
                    rowrange = np.arange(o_start, o_end, dtype=int)
                    ncols = colrange.size
                    nrows = rowrange.size

                    for i, row in enumerate(rowrange):
                        rows[i*ncols:(i+1)*ncols] = np.full(ncols, row, dtype=int)
                        cols[i*ncols:(i+1)*ncols] = colrange

            subJinfo.append((irow, icol), rows, cols, subjac, idxs)


        diag_end = -1  # keep track of last index where diagonal is filled

        # now iterate over the subjac info sorted in row major order
        for (irow, icol), o_start, o_end, i_start, i_end, subjac, idxs in \
                                          sorted(subJinfo, key=lambda x: x[0]):

                idxs, jac = subJinfo[jpair]
                # if the current input is only connected to certain entries
                # in its source, we have to map the sub-jacobian to the
                # appropriate rows of the big jacobian.
                if idxs:
                    col_list = np.array(idxs)+i_start
                else:
                    col_list = numpy.arange(i_start, i_end)
                cols.append(col_list)

                nidxs = col_list.size
                for i in range(o_start, o_end):
                    rows.append(np.full(nidxs, i))

                if issparse(jac):
                    if isinstance(jac, scipy.sparse.coo.coo_matrix):
                        # error check that idxs match up?
                        data.append(jac.data)
                    elif isinstance(jac, scipy.sparse.csr.csr_matrix):
                        pass
                    else:
                        raise TypeError("Sparse jacobian matrix of type '%s' is not supported." %
                                         type(jac).__name__)
                else:
                    data.append(jac.flatten())

            # entry on diagonal we need to fill.  We can't just start with
            # an identity matrix because if we end up with duplicate
            # row,col data values, those values will be added together
            # during sparse matrix creation, giving us an incorrect J.
            elif iname == oname:
                tmp = numpy.arange(o_start, o_end)
                rows.extend(tmp)
                cols.extend(tmp)
                data.append(np.full(tmp.size, -1.0))



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

        return partials, cache


    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass


if __name__ == '__main__':

    J = DenseJacobian(unknowns)
