
import numpy as np
from scipy.sparse import csc_matrix, csc_matrix, coo_matrix, eye as sp_eye
from itertools import groupby

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

        # numpy views into data array keyed on (oname, iname)
        self._views = {}

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
    # def __init__(self, unknowns):
    #     super(SparseJacobian, self).__init__(unknowns)

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

        subJinfo = []
        diags = []
        data_size = 0

        for ovar, ivar, subjac, idxs in self._sub_jac_iter(group, connections, prom_map):

            irowblock, (o_start, o_end) = self._ordering[ovar]
            icolblock, (i_start, i_end) = self._ordering[ivar]

            rows = np.empty(subjac.size, dtype=int)
            cols = np.empty(subjac.size, dtype=int)

            if issparse(subjac):
                # make sure it's in COO format.
                subjac = subjac.tocoo()

                # if the current input is only connected to certain entries
                # in its source, we have to map the sub-jacobian to the
                # appropriate rows of the big jacobian.
                if idxs:
                    cols[:] = subjac.col
                    cols += i_start

                    assert len(idxs) == subjac.shape[0]

                    for i, row in enumerate(subjac.row):
                        # note that if idxs are not in sorted order then row idxs
                        # won't be in sorted order
                        rows[i] = row + o_start + idxs[i]
                else:
                    rows[:] = subjac.row
                    rows += o_start
                    cols[:] = subjac.col
                    cols += i_start
            else:
                if idxs:
                    rowrange = np.array(idxs, dtype=int) + o_start
                else:
                    rowrange = np.arange(o_start, o_end, dtype=int)

                colrange = np.arange(i_start, i_end, dtype=int)
                ncols = colrange.size

                for i, row in enumerate(rowrange):
                    rows[i*ncols:(i+1)*ncols] = np.full(ncols, row, dtype=int)
                    cols[i*ncols:(i+1)*ncols] = colrange

            data_size += subjac.size

            # same var for row and col, so a block diagonal entry
            # (this only happens with states, and we don't have to worry
            # about having src_indices with states.
            if ivar == ovar:
                diags.append(ivar)

            subJinfo.append((irowblock, icolblock), rows, cols, subjac, idxs)

        # add diagonal entries
        eye_cache = {}
        missing_diags = set(self._ordering).difference(diags)
        for d in missing_diags:
            iblock, (start, end) = self._ordering[d]
            sz = end-start
            if sz not in eye_cache:
                eye_cache[sz] = sp_eye(sz, format='coo')

            rows = numpy.arange(start, end, dtype=int)
            cols = rows.copy()

            data_size += sz

            subJinfo.append((iblock, iblock), rows, cols, eye_cache[sz], None)

        data = np.empty(data_size)

        # now iterate over the subjacs sorted in row major order
        for blockrow, blocks in groupby(sorted(subJinfo, key=lambda x: x[0],
                                     key=lambda x: x[0][0]):
            for (irowblock, icolblock), rows, cols, subjac, idxs in blocks:
                # now iterate over subrows of these blocks...

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
