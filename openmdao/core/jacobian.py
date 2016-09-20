
import numpy as np
from scipy.sparse import csc_matrix, csc_matrix, coo_matrix, issparse, eye as sp_eye
from itertools import groupby

from six.moves import range
from six import iteritems, itervalues
# REQs
# - pre-allocation during setup
# - components will be passed this object and can set sub jacobians into it
#    using  J['a', 'b'] style syntax
# - for dense jacobian w/dense sub-jacobian, component can operate directly
#   on subview of the full jacbian, so no data copying
# - compute index arrays to allow fast copying of sparse subjac data into
#   the full jac's data array

class Jacobian(object):
    """
    Args
    ----

    slices : iter of (varname, (start, end)) for all variables that will make
    up the jacobian.

    """
    def __init__(self, slices):
        ordered_slices = sorted((item for item in slices), key=lambda x: x[1][0])
        self._slices = {
            n : slice(start, end) for n, (start, end) in ordered_slices
        }
        
        # record block order and slice for each vector variable in unknowns
        self._ordering = {
            n : (i, slc) for i, (n, slc) in enumerate(ordered_slices)
        }
        
        # number of rows (and columns) in J
        self.jsize = np.sum(s[1]-s[0] for _, s in itervalues(self._ordering))

    def __getitem__(self, key):
        return self.partials[self._slices[key[0]], self._slices[key[1]]]

class DenseJacobian(Jacobian):
    def assemble(self):
        pass

    def __setitem__(self, key, value):
        pass


class SparseJacobian(Jacobian):
    def __init__(self, slices):
        super(SparseJacobian, self).__init__(slices)

        # index arrays into data array keyed on (oname, iname)
        self._idx_arrays = {}


    def assemble(self, subjac_iter, direction):
        """ Assemble and return a sparse array containing the Jacobian for
        the given group.

        Args
        ----
        subjac_iter : iter over sub-jacobians
            An iterator of sub-jacobians that make up the overall jacobian.
            sub-jacobians can be numpy arrays, scalars, or scipy COO sparse
            matrices.

        direction : string('fwd')
            Derivative direction, can be 'fwd' or 'rev'.

        """

        subJinfo = []
        diags = []
        data_size = 0

        for ovar, ivar, subjac, idxs in subjac_iter:  #self._sub_jac_iter(group, connections, prom_map):

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

                    nrows = subjac.shape[0]
                    sorted_idxs = np.argsort(idxs)
                    minidx = np.min(idxs)

                    for i in range(nrows):
                        idxarray = np.nonzero(subjac.row==sorted_idxs[i]-minidx)
                        rows[idxarray] = i + o_start + idxs[sorted_idxs[i]]
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

            subJinfo.append(((irowblock, icolblock), (ovar, ivar),
                            rows, cols, subjac, idxs))

        # add diagonal entries
        eye_cache = {}
        missing_diags = set(self._ordering).difference(diags)
        for d in missing_diags:
            iblock, (start, end) = self._ordering[d]
            sz = end-start
            if sz not in eye_cache:
                eye_cache[sz] = sp_eye(sz, format='coo')

            rows = np.arange(start, end, dtype=int)
            # we don't modify rows or cols, so we can use the same array for both
            cols = rows

            subJ = eye_cache[sz]
            
            data_size += subJ.size

            subJinfo.append(((iblock, iblock), (d,d), rows, cols, subJ, None))

        data = np.empty(data_size)

        # dict for looking up idx array for a given (oval, ival) pair
        self._idx_arrays = {}

        # sort blocks into row major order.  We sort them this way so that
        # later we can subsort all blocks in a given block row by actual
        # array row, and then use the fact that we started with them in
        # sorted order to be able to back out the index arrays we're looking
        # for.
        sorted_blocks = sorted(subJinfo, key=lambda x: x[0])

        full_rows = []
        full_cols = []

        start = end = 0
        
        # now iterate over the sorted blocks and group them by block row
        for blockrow, blockiter in groupby(sorted_blocks, key=lambda x: x[0][0]):

            blockrow_offset = end # offset into the data array of this block row
            
            blocks = list(blockiter) # need to do this since blocks is an iterator
            
            # take the row and col arrays from our row blocks and create
            # new arrays with all of them stacked together.
            block_sub_rows = [t[2] for t in blocks]
            block_sub_cols = [t[3] for t in blocks]

            sub_rows = np.hstack(block_sub_rows)
            sub_cols = np.hstack(block_sub_cols)

            # calculate index array that sorts the stacked row and col arrays
            # in row major order
            sub_sorted = np.lexsort((sub_cols, sub_rows))

            # now sort these back into ascending order (our original stacked order)
            # so we can then just extract the individual index arrays that will
            # map each block into the combined data array.
            idx_arrays = np.argsort(sub_sorted)

            rowstart = rowend = 0
            
            # now iterate one more time in order through the blocks in this block row
            # to extract the individual index arrays
            offset = self._ordering[blocks[0][1][0]][1][0]
            for _, key, rows, cols, subjac, _ in blocks:
                end += rows.size
                rowend += rows.size
                self._idx_arrays[key] = idxs = idx_arrays[rowstart:rowend]+blockrow_offset
                if issparse(subjac):
                    #data[start:end] = subjac.data
                    data[idxs] = subjac.data
                else:
                    #data[start:end] = subjac.flatten()
                    data[idxs] = subjac.flat
                start = end
                rowstart = rowend

            full_rows.append(sub_rows[sub_sorted])
            full_cols.append(sub_cols[sub_sorted])

        # these final row and col arrays will be in
        # sorted (row major) order.
        final_rows = np.hstack(full_rows)
        final_cols = np.hstack(full_cols)

        partials = coo_matrix((data, (final_rows, final_cols)),
                              shape=(self.jsize, self.jsize))

        # tocsr will not change the order of the data array from that of coo,
        # but if later we add tocsc option, we'll have to revisit this since
        # the order will change and our index arrays will then be wrong.
        partials = partials.tocsr()

        if direction == 'rev':
            # CSR.T results in CSC, but doesn't change the data array order
            partials = partials.T

        self.partials = partials

        return partials


    def __setitem__(self, key, value):
        pass



if __name__ == '__main__':

    J = DenseJacobian(unknowns)
