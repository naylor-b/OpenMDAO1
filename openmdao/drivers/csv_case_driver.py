"""
OpenMDAO driver that runs a list of cases pulled from a CSV file.
"""
import csv
import numpy

from openmdao.drivers.case_driver import CaseDriver

def _str2a(s):
    s = s.strip()
    for dim, ch in enumerate(s):
        if ch != '[':
            break

    if dim == 1:
        return numpy.array(eval(','.join(s.split()).replace("[,","[")))
    elif dim == 2:
        rows = s.split('\n')
        for i in range(len(rows)):
            rows[i] = ','.join(rows[i].split()).replace("[,","[")
        return numpy.array(eval(','.join(rows)))
    elif dim == 0:  # numpy scalar
        return numpy.array(eval(s))
    else:
        raise TypeError("Can't convert numpy array of %d dimensions to str" %
                        dim)
    return s

_str2valdict = {
    int : int,
    float : float,
    str : str,
    list : eval,
    numpy.ndarray : _str2a,
}

def _toval(typ, s):
    return _str2valdict[typ](s)


class CSVCaseDriver(CaseDriver):
    """OpenMDAO driver that runs a list of cases pulled from a CSV file.

    Args
    ----
    fname : str
        Name of CSV file containing cases.

    num_par_doe : int, optional
        The number of cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, fname, dialect='excel', num_par_doe=1,
                 load_balance=False, **fmtparams):
        super(CaseDriver, self).__init__(num_par_doe=num_par_doe,
                                         load_balance=load_balance)
        self.fname = fname
        self.dialect = dialect
        self.fmtparams = fmtparams

    def _build_runlist(self):
        """Yield cases from our CSV file."""

        with open(self.fname, "rb") as f:
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


def save_cases_as_csv(driver, csv_file):
    """
    Take the cases from any PredeterminedRunsDriver and save them to a csv
    file for later use with a CSVCaseDriver.

    Args
    ----

    driver : PredeterminedRunsDriver
        The driver to save the cases from.

    csv_file : str
        The name of the file to save the csv to.
    """
    with open(csv_file, "wb") as f:
        writer = csv.writer(f, dialect=self.dialect, **self.fmtparams)

        # write the header
        writer.writerow(self.fields)

        for case in driver._build_runlist():
            writer.writerow(v for _,v in case)
