import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq

st.set_page_config(page_title="MST Visualizer", page_icon="🌐", layout="wide")

# Graph stored in session_state
if "G" not in st.session_state:
    st.session_state.G = nx.Graph()


def draw_graph(G, highlight_edges=None, title="Graph"):
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    # Node + edge drawing
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax, node_size=600)

    # Draw all edges in gray first
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, ax=ax)

    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

    # Highlight MST edges when provided
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=4, edge_color='red', ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)


def kruskal_mst(G):
    edges = sorted(G.edges(data=True), key=lambda x: x[2].get('weight', 1))
    parent = {node: node for node in G.nodes()}
    rank = {node: 0 for node in G.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            else:
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1

    mst_edges = []
    steps = []

    for u, v, data in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, data.get('weight', 1)))
            steps.append(mst_edges.copy())

    cost = sum(w for (_, _, w) in mst_edges)
    return steps, mst_edges, cost


def prim_mst(G, start):
    if start not in G:
        return [], [], 0
    visited = {start}
    edges = [(G[start][v].get('weight', 1), start, v) for v in G.neighbors(start)]
    heapq.heapify(edges)

    mst_edges = []
    steps = []

    while edges:
        w, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, w))
            steps.append(mst_edges.copy())
            for x in G.neighbors(v):
                if x not in visited:
                    heapq.heappush(edges, (G[v][x].get('weight', 1), v, x))

    cost = sum(w for (_, _, w) in mst_edges)
    return steps, mst_edges, cost


def bfs(G, start):
    if start not in G:
        return []
    visited = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(list(G.neighbors(node)))
    return visited


def dfs(G, start, visited=None):
    if start not in G:
        return []
    if visited is None:
        visited = []
    visited.append(start)
    for neighbor in G.neighbors(start):
        if neighbor not in visited:
            dfs(G, neighbor, visited)
    return visited


st.title("🌐 Minimum Spanning Tree Visualizer (Prim & Kruskal)")

with st.sidebar:
    st.header("Graph Controls")

    action = st.selectbox("Choose Action", ["Add Node", "Add Edge", "Delete Node", "Random Graph"])

    if action == "Add Node":
        node = st.text_input("Node Name", key="add_node")
        if st.button("Add Node"):
            if node:
                st.session_state.G.add_node(node)
                st.success(f"Added node '{node}'")

    elif action == "Add Edge":
        u = st.text_input("Node 1 (u)", key="edge_u")
        v = st.text_input("Node 2 (v)", key="edge_v")
        w = st.number_input("Weight", min_value=0.0, value=1.0, step=0.1, key="edge_w")
        if st.button("Add / Update Edge"):
            if u and v:
                st.session_state.G.add_edge(u, v, weight=float(w))
                st.success(f"Added/Updated edge {u} — {v} (w={w})")

    elif action == "Delete Node":
        node = st.text_input("Node to Delete", key="del_node")
        if st.button("Delete Node"):
            if node in st.session_state.G:
                st.session_state.G.remove_node(node)
                st.success(f"Removed node '{node}'")
            else:
                st.error("Node not in graph")

    elif action == "Random Graph":
        n = st.slider("Number of nodes", 4, 12, 6, key="rand_n")
        p = st.slider("Edge probability (%)", 10, 90, 40, key="rand_p") / 100.0
        if st.button("Generate Random Graph"):
            nodes = [str(i) for i in range(n)]
            H = nx.erdos_renyi_graph(n, p, seed=42)
            Gnew = nx.Graph()
            for i in range(n):
                Gnew.add_node(nodes[i])
            for a, b in H.edges():
                Gnew.add_edge(nodes[a], nodes[b], weight=round(1 + 9 * (hash((a, b)) % 100) / 100.0, 1))
            st.session_state.G = Gnew
            st.success("Random graph generated")

    st.markdown("---")
    st.write("### Traversal")
    start_node = st.text_input("Start Node for Traversal", key="trav_start")
    if st.button("Run BFS"):
        order = bfs(st.session_state.G, start_node)
        st.write("BFS order:", " → ".join(order) if order else "No nodes")
    if st.button("Run DFS"):
        order = dfs(st.session_state.G, start_node)
        st.write("DFS order:", " → ".join(order) if order else "No nodes")

st.write("### 📌 Current Graph")
draw_graph(st.session_state.G, title="Original Graph")

st.write("---")
st.write("### 🟢 Run Minimum Spanning Tree Algorithms")

col1, col2 = st.columns(2)

# Kruskal
with col1:
    st.subheader("Kruskal's Algorithm")
    if st.button("Run Kruskal's Algorithm"):
        steps, mst, cost = kruskal_mst(st.session_state.G)
        if not mst:
            st.info("MST is empty (graph disconnected or no edges).")
        else:
            st.write("#### Step-by-Step Execution:")
            for i, step in enumerate(steps):
                with st.expander(f"Step {i+1}: Added edge {step[-1][0]} ↔ {step[-1][1]} (weight: {step[-1][2]})"):
                    st.write(f"**Edges in MST so far:** {[(a, b, w) for (a, b, w) in step]}")
                    draw_graph(st.session_state.G, highlight_edges=[(u, v) for (u, v, _) in step], 
                             title=f"Kruskal's Algorithm - Step {i+1}")
            
            st.markdown("---")
            st.success("### ✅ Final MST Result (Kruskal's Algorithm)")
            
            # Show final MST visualization
            draw_graph(st.session_state.G, highlight_edges=[(u, v) for (u, v, _) in mst],
                      title="Final Minimum Spanning Tree (Kruskal)")
            
            # Display MST edges in a clean format
            st.write("#### 📋 MST Edges (in order of selection):")
            for i, (u, v, w) in enumerate(mst, 1):
                st.write(f"{i}. **{u} ↔ {v}** (weight: {w})")
            
            # Show path representation
            path_str = " → ".join([f"{u}—{v}({w})" for u, v, w in mst])
            st.write(f"#### 🛤️ MST Path Representation:")
            st.code(path_str, language="text")
            
            st.write(f"#### 💰 **Total Minimum Cost: {cost}**")
            st.write(f"#### 📊 **Total Edges in MST: {len(mst)}**")

# Prim
with col2:
    st.subheader("Prim's Algorithm")
    prim_start = st.text_input("Start Node for Prim", key="prim_start")
    if st.button("Run Prim's Algorithm"):
        if not prim_start:
            st.error("Enter a start node for Prim")
        else:
            steps, mst, cost = prim_mst(st.session_state.G, prim_start)
            if not mst:
                st.info("MST is empty (graph disconnected or invalid start).")
            else:
                st.write("#### Step-by-Step Execution:")
                for i, step in enumerate(steps):
                    with st.expander(f"Step {i+1}: Added edge {step[-1][0]} ↔ {step[-1][1]} (weight: {step[-1][2]})"):
                        st.write(f"**Edges in MST so far:** {[(a, b, w) for (a, b, w) in step]}")
                        draw_graph(st.session_state.G, highlight_edges=[(u, v) for (u, v, _) in step],
                                 title=f"Prim's Algorithm - Step {i+1}")
                
                st.markdown("---")
                st.success("### ✅ Final MST Result (Prim's Algorithm)")
                
                # Show final MST visualization
                draw_graph(st.session_state.G, highlight_edges=[(u, v) for (u, v, _) in mst],
                          title="Final Minimum Spanning Tree (Prim)")
                
                # Display MST edges in a clean format
                st.write("#### 📋 MST Edges (in order of selection):")
                for i, (u, v, w) in enumerate(mst, 1):
                    st.write(f"{i}. **{u} ↔ {v}** (weight: {w})")
                
                # Show path representation
                path_str = " → ".join([f"{u}—{v}({w})" for u, v, w in mst])
                st.write(f"#### 🛤️ MST Path Representation:")
                st.code(path_str, language="text")
                
                st.write(f"#### 💰 **Total Minimum Cost: {cost}**")
                st.write(f"#### 📊 **Total Edges in MST: {len(mst)}**")
                st.info(f"Started from node: **{prim_start}**")
