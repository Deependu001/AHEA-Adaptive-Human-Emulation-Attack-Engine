import json
import networkx as nx
import matplotlib.pyplot as plt

# Load saved attack graph
with open("attack_graph.json") as f:
    data = json.load(f)

G = nx.DiGraph()

# Add nodes
for node in data["nodes"]:
    G.add_node(node)

# Add edges with weights
for key, e in data["edges"].items():
    u, v = key.split("|")
    w = e["w"]
    color = "green" if w > 0.5 else "yellow" if w > 0.2 else "red"
    G.add_edge(u, v, weight=w, color=color)

# Draw graph
pos = nx.spring_layout(G, seed=42)
edges = G.edges()
colors = [G[u][v]["color"] for u,v in edges]

nx.draw(G, pos, with_labels=True, node_color="lightblue",
        edge_color=colors, arrows=True, font_size=8)
plt.title("AHEA Attack Graph (Evolving Paths)")
plt.show()