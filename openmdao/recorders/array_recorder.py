import sys

from itertools import chain

from six import string_types, iteritems

from openmdao.core.mpi_wrap import MPI
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

class ArrayRecorder(BaseRecorder):
    """ Recorder that saves cases as rows in a numpy array.
    """

    def __init__(self, arr=None):
        super(ArrayRecorder, self).__init__()

        self.options['record_unknowns'] = True
        self.options['record_params'] = False
        self.options['record_metadata'] = False
        self.options['record_resids'] = False
        self.options['record_derivs'] = False

        self._parallel = False
        self.arr = arr

    def startup(self, group):
        super(ArrayRecorder, self).startup(group)
        self._idx_map = { n: i for i,n in enumerate(self.options['includes'])}
        self.iter_count = 0

    def record_iteration(self, params, unknowns, resids, metadata):
        """Record the given run data in memory.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        iteration_coordinate = metadata['coord']

        if self.arr is None:
            raise RuntimeError("storage array was not set")

        row = self.arr[self.iter_count]

        success = metadata['success']
        nan = float('nan')

        idxs = self._idx_map
        for name, val in self._filter_vector(unknowns, 'u', iteration_coordinate):
             row[idxs[name]] = val if success else nan

        self.iter_count += 1

    def record_metadata(self, group):
        pass

    def record_derivatives(self, derivs, metadata):
        pass
