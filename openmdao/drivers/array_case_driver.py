"""
OpenMDAO driver that runs a sequence of cases specified by an array of size
(ncases x nparams).  Rows represent cases and columns contain the values of
specific variables for that case.
"""
import numpy

from six import string_types
from six.moves import range, zip

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from openmdao.recorders.array_recorder import ArrayRecorder

class ArrayCaseDriver(PredeterminedRunsDriver):
    """OpenMDAO driver that runs a sequence of cases based on values in
    an input 2D array.

    Args
    ----
    arr : numpy.ndarray
        An array of variable values where each row represents a case.

    colvars : list of str
        A list of variable names. The length of the list must match
        the number of columns in arr.

    num_par_doe : int, optional
        The number of cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, num_par_doe=1, load_balance=False):
        super(ArrayCaseDriver, self).__init__(num_par_doe=num_par_doe,
                                              load_balance=load_balance)
        self.desvar_array = None
        self.response_array = None
        self._respvars = []

    def _reset(self):
        dshape = self.desvar_array.shape

        # make sure response_array is shaped correctly relative to value of desvar_array
        if self.response_array is None or self.response_array.shape != (dshape[0],
                                                                        len(self._respvars)):
            self.response_array = numpy.empty((dshape[0], len(self._respvars)))

        self.response_array[:] = float('nan')

        # set the recorder array
        if self._full_comm.rank == 0:
            self._resp_recorder.arr = self.response_array
            self._resp_recorder.iter_count = 0

    def _check_desvar_array(self):
        if self.desvar_array is None:
            raise RuntimeError("desvar_array has not been set.")

        if not isinstance(self.desvar_array, numpy.ndarray):
            raise TypeError("desvar_array must be a numpy array.")

        shape = self.desvar_array.shape
        assert len(shape)==2, "desvar_array must be 2D, but shape is %s" % list(shape)
        assert shape[1] == len(self._desvars), "number of design vars must match number of desvar_array columns."

    def _setup(self):
        self._resp_recorder = ArrayRecorder()
        self._resp_recorder.options['includes'] = list(self._respvars)
        self.add_recorder(self._resp_recorder)

        super(ArrayCaseDriver, self)._setup()

    def add_response(self, name):
        """Add a variable(s) whose value will be collected after the execution
        of each case.

        Args
        ----

        name : str or iter of str
            The name of the response variable, or a list of names.
        """
        if isinstance(name, string_types):
            self._respvars.append(name)
        else:
            self._respvars.extend(name)

    def _build_runlist(self):
        """Yield cases from our sequence of cases."""

        self._check_desvar_array()
        self._reset()

        for row in range(self.desvar_array.shape[0]):
            yield ((n, v) for n,v in zip(self._desvars, self.desvar_array[row]))
