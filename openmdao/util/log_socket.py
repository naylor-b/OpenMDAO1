# this is from https://docs.python.org/2/howto/logging-cookbook.html

# Run this file to start a log server.  Setting the environment var OPENMDAO_SOCKET_LOGGING=1
# in any desired test processes will send their log messages to this server for display
# on the console where this server is running.

import os
import pickle
import logging
import logging.handlers
from six.moves import socketserver
import struct


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)

class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """

    allow_reuse_address = 1

    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort


SOCK_LOGGER = None

def enable_socket(level=logging.DEBUG,
                  port=logging.handlers.DEFAULT_TCP_LOGGING_PORT):
    global SOCK_LOGGER
    if SOCK_LOGGER is None:
        rootLogger = logging.getLogger('')
        rootLogger.setLevel(level)
        SOCK_LOGGER = logging.handlers.SocketHandler('localhost', port)
        # don't bother with a formatter, since a socket handler sends
        # the event as an unformatted pickle
        rootLogger.addHandler(SOCK_LOGGER)

def disable_socket():
    """ Stop sending log msgs to the log server. """
    global SOCK_LOGGER
    logging.getLogger().removeHandler(SOCK_LOGGER)
    SOCK_LOGGER = None

env_socket = os.environ.get('OPENMDAO_LOG_SOCKET')
if env_socket:
    port = int(env_socket)
    if port == 0:
        port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
    enable_socket(port=port)

def main():
    logging.basicConfig(
        format='%(relativeCreated)5d %(name)-8s %(levelname)-8s %(message)s')
    tcpserver = LogRecordSocketReceiver()
    print('Starting log server...')
    tcpserver.serve_until_stopped()

if __name__ == '__main__':
    main()
