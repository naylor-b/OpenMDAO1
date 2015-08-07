import os
import sys
import shutil
import json
import tempfile
import threading
import time
import pprint

import webbrowser
import SimpleHTTPServer
import SocketServer

from networkx.readwrite.json_graph import node_link_data


def _launch_browser(port, fname):
    time.sleep(1)
    webbrowser.get().open('http://localhost:%s/%s' % (port,fname))

def _startThread(fn):
    thread = threading.Thread(target=fn)
    thread.setDaemon(True)
    thread.start()
    return thread

def view_plot(tree, d3page='circlepack.html', port=8001):
    try:
        startdir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        with open('__graph.json', 'w') as f:
            #f.write("__mygraph__json = ")
            json.dump(tree, f)
            #f.write(";\n")

        httpd = SocketServer.TCPServer(("localhost", port),
                           SimpleHTTPServer.SimpleHTTPRequestHandler)

        print("starting server on port %d" % port)

        serve_thread  = _startThread(httpd.serve_forever)
        launch_thread = _startThread(lambda: _launch_browser(port, d3page))

        while serve_thread.isAlive():
            serve_thread.join(timeout=1)

    finally:
        try:
            os.remove('__graph.json')
        except:
            pass
        os.chdir(startdir)

def _to_id(name):
    """Convert a given name to a valid html id, replacing
    dots with hyphens."""
    return name.replace('.', '-')

def plot_sys_tree(system, d3page=''):
    """Open up a display of the System tree in a browser."""
    pass

def plot_graph(graph, excludes=(), d3page='fixedforce.html'):
    """Open up a display of the graph in a browser window."""

    tmpdir = tempfile.mkdtemp()
    fdir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(fdir, 'd3.min.js'), tmpdir)
    shutil.copy(os.path.join(fdir, d3page), tmpdir)

    data = node_link_data(graph)
    tmp = data.get('graph', [])
    data['graph'] = [dict(tmp)]

    startdir = os.getcwd()
    os.chdir(tmpdir)
    try:
        # write out the json as a javascript var
        # so we we're not forced to start our own webserver
        # to avoid cross-site issues
        with open('__graph.js', 'w') as f:
            f.write("__mygraph__json = ")
            json.dump(data, f)
            f.write(";\n")

        # open URL in web browser
        wb = webbrowser.get()
        wb.open('file://'+os.path.join(tmpdir, d3page))
    except Exception as err:
        print str(err)
    finally:
        os.chdir(startdir)
        print "remember to remove temp directory '%s'" % tmpdir
        # time.sleep(5) # sleep to give browser time
                       # to read files before we remove them
        # shutil.rmtree(tmpdir)
        # print "temp directory removed"
