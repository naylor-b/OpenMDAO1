
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
    logging.getLogger().removeHandler(CONSOLE)

if int(os.environ.get('OPENMDAO_ENABLE_CONSOLE', '0')):
    enable_console()

# this is from https://docs.python.org/2/howto/logging-cookbook.html
def enable_socket_logging(level=logging.DEBUG):
    rootLogger = logging.getLogger('')
    rootLogger.setLevel(level)
    socketHandler = logging.handlers.SocketHandler('localhost',
                        logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    # don't bother with a formatter, since a socket handler sends the event as
    # an unformatted pickle
    rootLogger.addHandler(socketHandler)

if int(os.environ.get('OPENMDAO_SOCKET_LOGGING', '0')):
    enable_socket_logging()
