""" Class definition for the Branch_and_Bound driver. This driver can be run
standalone or plugged into the AMIEGO driver.

This is the branch and bound algorithm that maximizes the constrained
expected improvement function and returns an integer infill point. The
algorithm uses the relaxation techniques proposed by Jones et.al. on their
paper on EGO,1998. This enables the algorithm to use any gradient-based
approach to obtain a global solution. Also, to satisfy the integer
constraints, a new branching scheme has been implemented.

Developed by Satadru Roy
School of Aeronautics & Astronautics
Purdue University, West Lafayette, IN 47906
July, 2016
Implemented in OpenMDAO, Aug 2016, Kenneth T. Moore
"""

from __future__ import print_function

from six import iteritems
from six.moves import range

import numpy as np

from openmdao.core.driver import Driver
from openmdao.surrogate_models.kriging import KrigingSurrogate
from openmdao.util.record_util import create_local_meta, update_local_meta


class Branch_and_Bound(Driver):
    """ Class definition for the Branch_and_Bound driver. This driver can be run
    standalone or plugged into the AMIEGO driver.
    
    This is the branch and bound algorithm that maximizes the constrained
    expected improvement function and returns an integer infill point. The
    algorithm uses the relaxation techniques proposed by Jones et.al. on
    their paper on EGO,1998. This enables the algorithm to use any
    gradient-based approach to obtain a global solution. Also, to satisfy the
    integer constraints, a new branching scheme has been implemented.
    """
    
    def __init__(self):
        """Initialize the Branch_and_Bound driver."""

        super(Branch_and_Bound, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = False
        self.supports['mixed_integer'] = False

        # Default surrogate. User can slot a modified one, but it essentially
        # has to provide what Kriging provides.
        self.surrogate = KrigingSurrogate

    