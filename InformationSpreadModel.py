import networkx as nx
import csv
import random

random.seed(3407)

def extract_connected_subgraph(graph, target_size=300, sparsity=0.8):
    """Extract a sparse and unevenly connected subgraph from the provided graph."""
    print("Extracting connected subgraph...")
    nodes = list(graph.nodes())
    weights = [deg for _, deg in graph.degree()]
    selected_nodes = random.choices(nodes, weights=weights, k=target_size)
    
    subgraph = nx.Graph(graph.subgraph(selected_nodes))
    edges_to_remove = random.sample(list(subgraph.edges()), int(len(subgraph.edges()) * (1 - sparsity)))
    subgraph.remove_edges_from(edges_to_remove)
    
    ensure_graph_connectivity(subgraph)
    print("Subgraph extraction complete.")
    return subgraph

def ensure_graph_connectivity(graph):
    """Ensure that the graph is connected, adding edges if necessary."""
    if not nx.is_connected(graph):
        print("Subgraph is not connected, adding edges...")
        for component in nx.connected_components(graph):
            connect_components(graph, component)

def connect_components(graph, component):
    """Connect disconnected components in the graph."""
    connected_components = list(nx.connected_components(graph))
    while len(connected_components) > 1:
        first_component = connected_components.pop()
        second_component = connected_components[0]
        graph.add_edge(random.choice(list(first_component)), random.choice(list(second_component)))

def simulate_threshold_model(graph, commitment_threshold, memory_size, num_rounds=1000):
    """Simulate the threshold model on the graph."""
    print(f"Model parameters: C={commitment_threshold}, M={memory_size}, T={num_rounds}")
    total_nodes = len(graph.nodes())
    memory_state = {node: ['A'] * memory_size for node in graph.nodes()}
    committed_nodes = set(random.sample(list(graph.nodes()), int(commitment_threshold * total_nodes)))
    
    simulate_information_spread(graph, committed_nodes, memory_state, memory_size, num_rounds)
    return calculate_conversion_rate(graph, committed_nodes, memory_state, memory_size)

def simulate_information_spread(graph, committed_nodes, memory_state, memory_size, num_rounds):
    """Simulate the spread of information across the graph."""
    num_edges = len(graph.edges())
    num_nodes = len(graph.nodes())
    sample_size = min(num_edges, num_nodes)  # Ensure the sample size does not exceed the number of edges
    for _ in range(num_rounds):
        for edge in random.sample(list(graph.edges()), sample_size):
            speaker, hearer = random.choice([(edge[0], edge[1]), (edge[1], edge[0])])
            propagate_information(speaker, hearer, memory_state, memory_size, committed_nodes)

def propagate_information(speaker, hearer, memory_state, memory_size, committed_nodes):
    """Propagate information from speaker to hearer based on their memory state."""
    message = 'A' if memory_state[speaker].count('B') <= memory_size / 2 else 'B'
    if hearer not in committed_nodes:
        memory_state[hearer].pop(0)
        memory_state[hearer].append(message)

def calculate_conversion_rate(graph, committed_nodes, memory_state, memory_size):
    """Calculate the conversion rate of non-committed nodes."""
    non_committed_nodes = set(graph.nodes()) - committed_nodes
    converted_count = sum(1 for node in non_committed_nodes if memory_state[node].count('B') > memory_size / 2)
    return converted_count / len(non_committed_nodes)

def load_network_data(file_path):
    """Load network data from a CSV file."""
    graph = nx.Graph()
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            graph.add_edge(int(row[0]), int(row[1]))
    return graph

def main():
    print("Loading network data...")
    graph = load_network_data("./deezer_clean_data/RO_edges.csv")
    print("Network data loaded.")

    M = 21
    subgraph = extract_connected_subgraph(graph)
    results = []
    print("Starting parameter search simulation...")
    for m in range(1, M):
        result = process_m_value((subgraph.copy(), m))
        if result:
            results.append(result)
    print("Simulation parameter search complete.")
    
    print("Saving results to file...")
    save_results_to_file('result_1000.csv', results)
    print("Results saved.")

def process_m_value(params):
    graph, m = params
    print("Starting simulation for subgraph...")
    for c in range(10, 100):
        p = simulate_threshold_model(graph, c/100, m)
        if p == 1:
            print(f"Threshold model parameter found: C={c/100}, M={m}")
            return m, c/100
    print("No suitable threshold model parameter found.")
    return None

def save_results_to_file(filename, results):
    """Save the simulation results to a CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['M', 'C'])
        for result in results:
            writer.writerow(result)

if __name__ == '__main__':
    main()
