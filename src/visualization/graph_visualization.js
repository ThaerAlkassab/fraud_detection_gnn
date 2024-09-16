// graph_visualization.js
const width = 800, height = 600;
const svg = d3.select("svg")
              .attr("width", width)
              .attr("height", height);

const graphData = {
    nodes: [
        { id: 0, fraud: true, age: 40 },
        { id: 1, fraud: false, age: 30 },
        // Add more nodes based on your dataset
    ],
    links: [
        { source: 0, target: 1 },  // Edge linking node 0 and node 1
        // Add more links based on relationships (e.g., common transactions)
    ]
};

const simulation = d3.forceSimulation(graphData.nodes)
    .force("link", d3.forceLink(graphData.links).distance(100))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2));

const link = svg.selectAll("line")
    .data(graphData.links)
    .enter().append("line")
    .style("stroke", "#aaa");

const node = svg.selectAll("circle")
    .data(graphData.nodes)
    .enter().append("circle")
    .attr("r", 10)
    .attr("fill", d => d.fraud ? "red" : "blue");

simulation.on("tick", () => {
    link.attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    node.attr("cx", d => d.x)
        .attr("cy", d => d.y);
});
