"""
OpenMDAO driver that runs a set of cases from a pickle file.
"""
from six.moves import cPickle as pickle

from openmdao.drivers.case_driver import CaseDriver


class PickleCaseDriver(CaseDriver):
    """OpenMDAO driver that runs a sequence of cases from a pickle file.

    Args
    ----
    fname : str
        Name of pickle file containing cases.

    num_par_doe : int, optional
        The number of cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, fname, num_par_doe=1, load_balance=False):
        super(PickleCaseDriver, self).__init__(num_par_doe=num_par_doe,
                                               load_balance=load_balance)
        self.fname = fname

    def _build_runlist(self):
        """Yield cases from our sequence of cases."""

        with open(self.fname, "rb") as f:
            self.cases = pickle.load(f)

        for case in self.cases:
            yield case


def save_cases_as_pickle(driver, pickle_file):
    """
    Take the cases from any PredeterminedRunsDriver and save them to a pickle
    file for later use with a PickleCaseDriver.

    Args
    ----

    driver : PredeterminedRunsDriver
        The driver to save the cases from.

    pickle_file : str
        The name of the file to save the pickle to.
    """
    with open(pickle_file, "wb") as f:
        pickle.dump([list(case) for case in driver._build_runlist()], f, -1)
