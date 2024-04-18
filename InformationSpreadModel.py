import networkx as nx
import csv
import random
from multiprocessing import Pool, cpu_count

random.seed(3407)

def sample_subgraph(G, size=100):
    """基于节点度的抽样方法抽取子网络"""
    degrees = nx.degree(G)
    # 根据度排序，选择度最高的节点
    sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, d in sorted_nodes[:size]]
    # 从选定的节点生成子图
    H = G.subgraph(top_nodes)
    return H

def threshold_model(G, C, M, T=1000):
    N = len(G.nodes())
    node_memory = {node: ['A']*M for node in G.nodes()}
    all_nodes = set(G.nodes())
    committed_nodes = set(random.sample(list(G.nodes()), int(C * N)))
    non_committed_nodes = all_nodes - committed_nodes
    for node in committed_nodes:
        node_memory[node] = ['B']*M
    for round in range(T):
        edges = [random.choice(list(G.edges())) for _ in range(N)]
        for edge in edges:
            speaker, hearer = random.choice([(edge[0], edge[1]), (edge[1], edge[0])])
            speak = 'A'
            if node_memory[speaker].count('B') > M/2:
                speak = 'B'
            if hearer not in committed_nodes:
                node_memory[hearer].pop(0)
                node_memory[hearer].append(speak)
    non_committed_count = len(non_committed_nodes)
    x = 0
    for i in list(non_committed_nodes):
        if node_memory[i].count('B') > M/2:
            x += 1
    return x / non_committed_count

def process_m_value(params):
    G, m = params
    sub_G = sample_subgraph(G)  # 抽取子网络
    for c in range(10, 100):
        p = threshold_model(sub_G, c/100, m)
        if p == 1:
            return m, c/100
    return None

if __name__ == '__main__':
    G = nx.Graph()
    filePath = "./deezer_clean_data/RO_edges.csv"
    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过头部信息
        for row in reader:
            G.add_edge(int(row[0]), int(row[1]))
    M = 21
    pool = Pool(4)
    params_list = [(G.copy(), m) for m in range(1, M)]
    results = pool.map(process_m_value, params_list)
    pool.close()
    pool.join()
    with open('result.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['M', 'C'])
        for result in results:
            if result is not None:
                writer.writerow(result)
