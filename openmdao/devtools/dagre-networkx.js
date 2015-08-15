
(function(){
    dagre.networkx = {};

    dagre.networkx.create_graph = function(graph) {
        // Create a new directed graph
        var g = new dagreD3.graphlib.Graph().setGraph({});

        // Add states to the graph, set labels, and style
        graph.nodes.forEach(function(node) {
          node.rx = node.ry = 5;
          g.setNode(node.label, node);
        });

        graph.links.forEach(function(link) {
            g.setEdge(link.src, link.tgt, {});
        });

        return g;
    };

    dagre.networkx.render = function(g) {
        // Create the renderer
        var render = new dagreD3.render();

        // Set up an SVG group so that we can translate the final graph.
        var svg = d3.select("svg"),
            inner = svg.append("g");

        // Set up zoom support
        var zoom = d3.behavior.zoom().on("zoom", function() {
            inner.attr("transform", "translate(" + d3.event.translate + ")" +
                                        "scale(" + d3.event.scale + ")");
          });
        svg.call(zoom);

        // Run the renderer. This is what draws the final graph.
        render(inner, g);

        // Center the graph
        var initialScale = 0.75;
        zoom
          .translate([(svg.attr("width") - g.graph().width * initialScale) / 2, 20])
          .scale(initialScale)
          .event(svg);
        svg.attr('height', g.graph().height * initialScale + 40);
    };
});
