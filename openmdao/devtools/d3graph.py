from __future__ import print_function

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

import networkx as nx
from networkx.readwrite.json_graph import node_link_data


def _launch_browser(port, fname):
    time.sleep(1)
    webbrowser.get().open('http://localhost:%s/%s' % (port,fname))

def _startThread(fn):
    thread = threading.Thread(target=fn)
    thread.setDaemon(True)
    thread.start()
    return thread

def add_graph_meta(group, graph):
    """
    Add metadata for name, etc. to graph nodes/edges for display
    using d3 graph viewers.
    """
    graph = nx.DiGraph(graph)
    graph.graph['title'] = group.pathname if group.pathname else 'top'

    udict = group._unknowns_dict
    pdict = group._params_dict

    for node, meta in graph.nodes_iter(data=True):
        if node in udict:
            meta['full'] = udict[node]['pathname']
            meta['short'] = udict[node]['promoted_name']
        elif node in pdict:
            meta['full'] = pdict[node]['pathname']
            meta['short'] = pdict[node]['promoted_name']
        else:
            meta['full'] = node
            if '.' in node:
                meta['short'] = node.rsplit('.', 1)[1]
            else:
                meta['short'] = node

    return graph

def view_tree(tree, d3page='collapse_tree.html', port=8001):
    """
    Args
    ----
    tree : nested dict
        A nested dictionary indictating the structure of the system tree. Leaf nodes_iter
        in the tree are variables and branch nodes are systems.

    d3page : str, optional
        The name of the html file used to view the tree.

    port : int, optional
        The port number for the web server that serves the tree viewing page.
    """
    try:
        startdir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        from pprint import pprint
        with open('__graph.json', 'w') as f:
            #f.write("__mygraph__json = ")
            pprint(tree, width=70)
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

def view_dagre(graph, port=8001):
    page = 'dagre.html'

    # get json version of graph
    g = {}
    dlist = []
    for node, data in graph.nodes_iter(data=True):
        dlist.append(data.copy())
        dlist[-1]['id'] = node
        dlist[-1]['label'] = node.rsplit('.', 1)[-1]
        if '.' in node:
            dlist[-1]['parent'] = node.rsplit('.', 1)[0]
    g['nodes'] = dlist

    dlist = []
    for u,v,data in graph.edges_iter(data=True):
        dlist.append(data.copy())
        dlist[-1]['src'] = u
        dlist[-1]['tgt'] = v
    g['links'] = dlist

    try:
        startdir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # write out the json as a javascript var
        # so we we're not forced to start our own webserver
        # to avoid cross-site issues
        with open('__graph.json', 'w') as f:
            json.dump(g, f)

        httpd = SocketServer.TCPServer(("localhost", port),
                           SimpleHTTPServer.SimpleHTTPRequestHandler)

        print("starting server on port %d" % port)

        serve_thread  = _startThread(httpd.serve_forever)
        launch_thread = _startThread(lambda: _launch_browser(port, page))

        while serve_thread.isAlive():
            serve_thread.join(timeout=1)

    finally:
        os.chdir(startdir)

def view_cyto(graph, port=8001, compound=False):
    page = 'cyto.html'

    # get json version of graph
    g = {}
    dlist = []
    for node, data in graph.nodes_iter(data=True):
        data = data.copy()
        data['id'] = node
        data['name'] = node.rsplit('.', 1)[-1]
        if compound and '.' in node:
            data['parent'] = node.rsplit('.', 1)[0]
        dlist.append({'data': data})
    g['nodes'] = dlist

    dlist = []
    for u,v,data in graph.edges_iter(data=True):
        data = data.copy()
        data['id'] = u+v
        data['source'] = u
        data['target'] = v
        dlist.append({'data': data})
    g['edges'] = dlist

    try:
        startdir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # write out the json as a javascript var
        # so we we're not forced to start our own webserver
        # to avoid cross-site issues
        with open('__graph.json', 'w') as f:
            json.dump(g, f)

        httpd = SocketServer.TCPServer(("localhost", port),
                           SimpleHTTPServer.SimpleHTTPRequestHandler)

        print("starting server on port %d" % port)

        serve_thread  = _startThread(httpd.serve_forever)
        launch_thread = _startThread(lambda: _launch_browser(port, page))

        while serve_thread.isAlive():
            serve_thread.join(timeout=1)

    finally:
        os.chdir(startdir)

def view_graph(group, d3page='fixedforce.html'):
    """Open up a display of the graph in a browser window."""

    tmpdir = tempfile.mkdtemp()
    fdir = os.path.dirname(os.path.abspath(__file__))
    d3dir = os.path.join(fdir, 'd3')
    shutil.copy(os.path.join(fdir, 'd3.js'), tmpdir)
    shutil.copy(os.path.join(d3dir, 'd3.layout.js'), tmpdir)
    #shutil.copy(os.path.join(fdir, 'packages.js'), tmpdir)
    shutil.copy(os.path.join(fdir, d3page), tmpdir)


    data = { '_subs': {} }

    graph = add_graph_meta(group, group._get_sys_graph())
    data.update(node_link_data(graph))
    tmp = data.get('graph', [])
    data['graph'] = [dict(tmp)]

    for g in group.subgroups(recurse=True):
        graph = add_graph_meta(g, g._get_sys_graph())
        data['_subs'][g.pathname] = node_link_data(graph)
        tmp = data['_subs'][g.pathname].get('graph', [])
        data['_subs'][g.pathname]['graph'] = [dict(tmp)]

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
        print (str(err))
    finally:
        os.chdir(startdir)
        print ("remember to remove temp directory '%s'" % tmpdir)
        # time.sleep(5) # sleep to give browser time
                       # to read files before we remove them
        # shutil.rmtree(tmpdir)
        # print "temp directory removed"
