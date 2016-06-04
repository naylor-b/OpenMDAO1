
import os
import sys
import logging, logging.handlers


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

SOCK_LOGGER = None

# this is from https://docs.python.org/2/howto/logging-cookbook.html
def enable_socket_logging(level=logging.DEBUG):
    global SOCK_LOGGER
    if SOCK_LOGGER is None:
        rootLogger = logging.getLogger('')
        rootLogger.setLevel(level)
        SOCK_LOGGER = logging.handlers.SocketHandler('localhost',
                            logging.handlers.DEFAULT_TCP_LOGGING_PORT)
        # don't bother with a formatter, since a socket handler sends the event as
        # an unformatted pickle
        rootLogger.addHandler(SOCK_LOGGER)

def disable_socket_log():
    """ Stop sending log msgs to the log server. """
    global SOCK_LOGGER
    logging.getLogger().removeHandler(SOCK_LOGGER)
    SOCK_LOGGER = None

env_socket = os.environ.get('OPENMDAO_LOG_SOCKET', '0')
if env_socket and int(env_socket):
    enable_socket_logging()
