
(function(){
    dagre_networkx = {};

    dagre_networkx.create_graph = function(graph) {
        var color = d3.scale.category20();

        var color_idx = 0;
        var colors = {};
        // first, figure out how many clusters we have and assign
        // different colors to them
        hasparent = false;
        graph.nodes.forEach(function(node) {
            if ( node.parent !== undefined ) {
                if ( colors[node.parent] == undefined ) {
                    colors[node.parent] = color(color_idx);
                    color_idx = color_idx + 1;
                    hasparent = true;
                }
            }
        });

        // Create a new directed graph
        if (hasparent) {
            var g = new dagreD3.graphlib.Graph({compound:true}).setGraph({});
        }
        else {
            var g = new dagreD3.graphlib.Graph().setGraph({});
        }

        // Add states to the graph, set labels, and style
        graph.nodes.forEach(function(node) {
          node.rx = node.ry = 5;
          if (hasparent && colors[node.id] !== undefined ) {
              node.clusterLabelPos = 'top';
              node.style = 'fill: '+colors[node.id];
          }
          g.setNode(node.id, node);
        });

        if (hasparent) {
            graph.nodes.forEach(function(node) {
                if ( node.parent !== undefined ) {
                    g.setParent(node.id, node.parent)
                }
            });
        }

        graph.links.forEach(function(link) {
            g.setEdge(link.src, link.tgt, {lineInterpolate: 'basis'});
        });

        return g;
    };

    dagre_networkx.render = function(g) {
        // Create the renderer
        var renderer = new dagreD3.render();
        //var renderer = new dagreD3.Renderer();
        var l = dagreD3.dagre.layout()
                      .nodeSep(100)
                      .rankSep(200)
                      .edgeSep(80)
                      .rankDir("LR");
        renderer.layout(l);

        // Override drawNodes to set up the click.
        var oldDrawNodes = renderer.drawNodes();
        renderer.drawNodes(function(g, svg) {
             var svgNodes = oldDrawNodes(g, svg);

             // on click event handler
             svgNodes.on('click', function(d) { console.log('Clicked on node - ' + d); });

             return svgNodes;
        });

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
        //render(inner, g);
        layout = renderer.run(g, inner);

        // Center the graph
        // var initialScale = 0.75;
        // zoom
        //   .translate([(svg.attr("width") - g.graph().width * initialScale) / 2, 20])
        //   .scale(initialScale)
        //   .event(svg);
        // svg.attr('height', g.graph().height * initialScale + 40);

        var xCenterOffset = (svg.attr('width') - layout.graph().width) / 2;
        svgGroup.attr('transform', 'translate(' + xCenterOffset + ', 80)');
        svg.attr('height', layout.graph().height + 200);
    };
})();
