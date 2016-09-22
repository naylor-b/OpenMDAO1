
import numpy as np
from scipy.sparse import csc_matrix, csc_matrix, coo_matrix, issparse, eye as sp_eye
from itertools import groupby
from copy import copy

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
    def __init__(self, slices, direction):
        ordered_slices = sorted((item for item in slices), key=lambda x: x[1][0])
        self._slices = {
            n : slice(start, end) for n, (start, end) in ordered_slices
        }

        # record block order and slice for each vector variable in slices
        self._ordering = {
            n : (i, slc) for i, (n, slc) in enumerate(ordered_slices)
        }

        # number of rows (and columns) in J
        self.jsize = np.sum(s[1]-s[0] for _, s in itervalues(self._ordering))

        self._fwd = (direction=='fwd')

    def __getitem__(self, key):
        if self._fwd:
            return self.partials[self._slices[key[0]], self._slices[key[1]]]
        else:
            return self.partials[self._slices[key[1]], self._slices[key[0]]]


class MVPJacobian(Jacobian):
    """ Build a dense jacobian by obtaining matrix vector products from the
    solver.

    Args
    ----
    jsize : int
        Number of rows (and also columns) in the jacobian.

    mult : function
        Solver mult function to coordinate the matrix vector product

    """

    def __init__(self, jsize, mult):
        ident_vec = np.zeros(jsize)
        self.partials = partials = np.empty((jsize, jsize))

        for i in range(jsize):
            ident_vec[i-1] = 0.0
            ident_vec[i] = 1.0
            partials[:, i] = mult(ident_vec)


class DenseJacobian(Jacobian):
    def __init__(self, slices, subjac_iter, direction):
        super(DenseJacobian, self).__init__(slices, direction)

        partials = -np.eye(self.jsize)
        
        self._src_indices = {}

        for ovar, ivar, subjac, idxs in subjac_iter:
            self._src_indices[(ovar, ivar)] = idxs

            irowblock, (o_start, o_end) = self._ordering[ovar]
            icolblock, (i_start, i_end) = self._ordering[ivar]

            if issparse(subjac):
                subjac = subjac.A

            if idxs:
                for count, j in enumerate(idxs):
                    partials[o_start:o_end, i_start+j] = subjac[:,count]
            else:
                partials[o_start:o_end, i_start:i_end] = subjac

        if self._fwd:
            self.partials = partials
        else:
            self.partials = partials.T

    def __setitem__(self, key, value):
        oname, iname = key
        
        if issparse(value):
            value = value.A

        idxs = self._src_indices[key]
        
        if idxs is None:
            if self._fwd:
                self.partials[self._slices[oname], self._slices[iname]] = value
            else:
                self.partials[self._slices[iname], self._slices[oname]] = value.T
        else:
            rslice = self._slices[oname]
            cstart = self._slices[iname].start
            partials = self.partials
            if self._fwd:
                for count, j in enumerate(idxs):
                    partials[rslice, cstart+j] = value[:,count]
            else:
                value = value.T
                for count, j in enumerate(idxs):
                    partials[cstart+j, rslice] = value[count, :]
                


class SparseJacobian(Jacobian):
    """ Build a sparse jacobian made up of either sparse or dense sub-jacobians.

    Args
    ----
    slices : iter over slices for each variable in the jacobian
        Yields tuples of the form (name, (start_index, end_index)) for each
        variable in the jacobian.

    subjac_iter : iter over sub-jacobians
        An iterator of sub-jacobians that make up the overall jacobian.
        sub-jacobians can be numpy arrays, scalars, or scipy COO sparse
        matrices. The iterator should return tuples of the form
        (ovar, ivar, subjac, src_indices) where src_indices can be
        None.

    direction : string('fwd')
        Derivative direction, can be 'fwd' or 'rev'.

    """
    def __init__(self, slices, subjac_iter, direction):
        super(SparseJacobian, self).__init__(slices, direction)

        # index arrays into data array keyed on (oname, iname)
        self._idx_arrays = {}
        
        self.sub_idxs = {} # accounts for reordering of cols due to src_indices

        subJinfo = []
        diags = []
        data_size = 0

        for ovar, ivar, subjac, idxs in subjac_iter:

            irowblock, (o_start, o_end) = self._ordering[ovar]
            icolblock, (i_start, i_end) = self._ordering[ivar]

            if issparse(subjac):
                # make sure it's in COO format.
                subjac = subjac.tocoo()

                # if the current input is only connected to certain entries
                # in its source, we have to map the sub-jacobian to the
                # appropriate cols of the big jacobian.
                if idxs:
                    cols = []
                    rows = []
                    
                    for count, idx in enumerate(idxs):
                        idxarray = np.nonzero(subjac.col==count)[0]
                        cols.append(np.array([idx]*idxarray.size)+i_start)
                        rows.append(subjac.row[idxarray]+o_start)
                        
                    rows = np.hstack(rows)
                    cols = np.hstack(cols)
                    
                    # keep track of how we have to transform our local data array
                    # based on the columns we changed
                    lex = np.lexsort((cols, rows))
                    self.sub_idxs[(ovar, ivar)] = lex
                    rows = rows[lex]
                    cols = cols[lex]
                else:
                    rows = subjac.row.copy()
                    rows += o_start
                    cols = subjac.col.copy()
                    cols += i_start
                    
                data_size += rows.size
            else:
                rowrange = np.arange(o_start, o_end, dtype=int)

                if idxs:
                    colrange = np.array(idxs, dtype=int) + i_start
                else:
                    colrange = np.arange(i_start, i_end, dtype=int)

                ncols = colrange.size

                rows = np.empty(rowrange.size*colrange.size, dtype=int)
                cols = np.empty(rowrange.size*colrange.size, dtype=int)

                for i, row in enumerate(rowrange):
                    rows[i*ncols: (i+1)*ncols] = np.full(ncols, row, dtype=int)
                    cols[i*ncols: (i+1)*ncols] = colrange

                data_size += subjac.size

            # same var for row and col, so a block diagonal entry
            # (this only happens with states, and we don't have to worry
            # about having src_indices with states.
            if ivar == ovar:
                diags.append(ivar)

            subJinfo.append(((irowblock, icolblock), (ovar, ivar),
                            rows, cols, subjac, idxs))

        # add diagonal entries.  We have to handle diagonal entries specially for a sparse
        # jacobian, because when constructing a sparse matrix, data with duplicated row and col
        # indices will be added together, and we don't want that.
        eye_cache = {}
        missing_diags = set(self._ordering).difference(diags)
        for d in self._ordering:
            if d not in diags:
                iblock, (start, end) = self._ordering[d]
                sz = end-start
                if sz not in eye_cache:
                    eye_cache[sz] = -sp_eye(sz, format='coo')
    
                rows = np.arange(start, end, dtype=int)
                # we don't modify rows or cols, so we can use the same array for both
                cols = rows
                data_size += sz
    
                subJinfo.append(((iblock, iblock), (d,d), rows, cols, eye_cache[sz], None))

        data = np.empty(data_size)

        # dict for looking up idx array for a given (oval, ival) pair
        self._idx_arrays = {}

        # sort blocks into row major order.  We sort them this way so that
        # later we can subsort all blocks in a given block row by actual
        # array row, and then use the fact that we started with them in
        # sorted order to be able to back out the index arrays we're looking
        # for. Later, when a component sets its sub-jacobian's values into the
        # big jacobian, we can use the index arrays to write the sub-jacobian
        # data values into the correct locations in the big jacobian's data array.
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
            for _, key, rows, cols, subjac, subidxs in blocks:
                end += rows.size
                rowend += rows.size
                self._idx_arrays[key] = idxs = idx_arrays[rowstart:rowend]+blockrow_offset
                if issparse(subjac):
                    if key in self.sub_idxs:
                        data[idxs] = subjac.data[self.sub_idxs[key]]
                    else:
                        data[idxs] = subjac.data
                else:
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
        self.partials = partials.tocsr()

        if direction == 'rev':
            # CSR.T results in CSC, but doesn't change the data array order
            self.partials = self.partials.T


    def __setitem__(self, key, value):
        if issparse(value):
            if key in self.sub_idxs:
                self.partials.data[self._idx_arrays[key]] = value.data[self.sub_idxs[key]]
            else:
                self.partials.data[self._idx_arrays[key]] = value.data
        else:
            self.partials.data[self._idx_arrays[key]] = value.flat
