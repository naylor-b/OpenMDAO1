"""
OpenMDAO driver that runs a list of cases pulled from a CSV file.
"""
import sys
import csv
import numpy

from six import next

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from openmdao.recorders.csv_recorder import serialize

# TODO: this (and our current CSV recorder) can't handle anything but
#       1 D arrays.  In order to handle higher dimensions we need to
#       store shape information in the string that gets written to the file.
def _str2a(s):
    return numpy.array([float(v) for v in s.strip().split(',')])

_str2valdict = {
    int : int,
    float : float,
    str : str,
    list : eval,
    numpy.ndarray : _str2a,
    numpy.float64 : numpy.float64,
}

def _toval(typ, s):
    return _str2valdict[typ](s)


class CSVCaseDriver(PredeterminedRunsDriver):
    """OpenMDAO driver that runs a list of cases pulled from a CSV file.

    Args
    ----
    fname : str
        Name of CSV file containing cases.

    dialect : str ('excel')
        Tells the CSV reader what dialect of CSV to use.

    num_par_doe : int, optional
        The number of cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, fname, dialect='excel', num_par_doe=1,
                 load_balance=False, **fmtparams):
        super(CSVCaseDriver, self).__init__(num_par_doe=num_par_doe,
                                            load_balance=load_balance)
        self.fname = fname
        self.dialect = dialect
        self.fmtparams = fmtparams

    def _build_runlist(self):
        """Yield cases from our CSV file."""

        if sys.version_info >= (3, 0, 0):
            mode = "r"
        else:
            mode = "rb"

        with open(self.fname, mode) as f:
            reader = csv.reader(f, dialect=self.dialect, **self.fmtparams)

            self.fields = None
            unknowns = self.root.unknowns
            params = self.root.params

            for row in reader:
                case = []
                if self.fields is None:
                    self.fields = row
                    self.types = []
                    for name in row:
                        if name in unknowns:
                            self.types.append(type(unknowns[name]))
                        else:
                            self.types.append(type(params[name]))
                    continue

                for i, val in enumerate(row):
                    case.append((self.fields[i], _toval(self.types[i], val)))

                yield case


def save_cases_as_csv(driver, csv_file, dialect='excel', **fmtparams):
    """
    Take the cases from a PredeterminedRunsDriver and save them to a csv
    file for later use with a CSVCaseDriver.  Note that all cases
    coming from the PredeterminedRunsDriver must have the same set of
    design variables.

    Args
    ----

    driver : PredeterminedRunsDriver
        The driver to save the cases from.

    csv_file : str
        The name of the file to save the csv to.

    fmtparams : dict
        Keyword formatting args to pass to csv.writer.
    """
    if sys.version_info >= (3, 0, 0):
        args = [csv_file, "w",]
        kwargs = {'newline': ''}
    else:
        args = [csv_file, "wb"]
        kwargs = {}

    with open(*args, **kwargs) as f:
        writer = csv.writer(f, dialect=dialect, **fmtparams)

        it = driver._build_runlist()

        try:
            case = list(next(it))

            # write the header
            writer.writerow([n for n,_ in case])

            while True:
                writer.writerow([serialize(v) for _, v in case])
                case = next(it)

        except StopIteration:
            pass
