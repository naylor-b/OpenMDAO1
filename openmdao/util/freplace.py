
import os
import sys
import logging
import traceback


def func_replace(mod_import, fname, replace=True):
    """
    Use this decorator to replace a function with a different
    function.  This can be used to replace a given function with a cython
    equivalent, a function with a debug version, etc.

    Args
    ----
    mod_import : str
        A string containing the module path for the module that contains
        the function definition.

    fname : str
        The name of the function within the specified module.

    replace : bool(True)
        If False, do not replace the decorated function.
    """
    def wrap(f):
        if replace:
            try:
                mod = __import__(mod_import, globals(), locals(), [fname])
            except ImportError:
                logging.warning(traceback.format_exc())
            else:
                f = getattr(mod, fname)

        return f

    return wrap
