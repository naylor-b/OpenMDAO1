
import os
import sys
import logging, logging.handlers
from openmdao.util.log_socket import enable_socket, disable_socket

# Optional handler which writes messages to sys.stderr
CONSOLE = None

def enable_console(level=logging.WARNING):
    """ Configure logging to receive log messages at the console. """
    global CONSOLE
    if CONSOLE is None:
        # define a Handler which writes messages to sys.stderr
        CONSOLE = logging.StreamHandler()
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(levelname)s %(name)s: %(message)s')
        # tell the handler to use this format
        CONSOLE.setFormatter(formatter)
    CONSOLE.setLevel(level)
    logging.getLogger().addHandler(CONSOLE)

def disable_console():
    """ Stop receiving log messages at the console. """
    global CONSOLE
    logging.getLogger().removeHandler(CONSOLE)
    CONSOLE = None

env_console = os.environ.get('OPENMDAO_LOG_CONSOLE')
if env_console and int(env_console):
    enable_console()
