import copy
import random
from collections import deque

import numpy as np
from scipy import sparse
import networkx as nx


class Algorithm:
    """图相关的算法

    """
    @staticmethod
    def sir_spread(adjacency_table, sources, beta):
        """Spread message by SIR model.

        Position arguments:
        adjacency_table -- 2D-list, represent a network
        sources -- list, spread sources
        beta -- spread rate

        Return:
        infected nodes
        """
        open_table = deque(sources)
        close_table = {node: True for node in open_table}

        while open_table:
            current_node = open_table.pop()
            for neighbor in adjacency_table[current_node]:
                if neighbor not in close_table:
                    if random.random() <= beta:
                        open_table.appendleft(neighbor)
                        close_table[neighbor] = True
        infected = list(close_table)
        return infected

    @staticmethod
    def breadth_priority_traversal(start_nodes: list, adjacency_table) -> list:
        """图的广度优先遍历
        """
        open = deque(start_nodes)
        close = {start_node: True for start_node in start_nodes}

        while open:
            current_node = open.popleft()  # 头出尾进
            for neighbor in adjacency_table[current_node]:
                if neighbor not in close:
                    open.append(neighbor)
                    close[neighbor] = True

        return list(close)

    @staticmethod
    def depth_priority_traversal(start_node, adjacency_table, depth=None,
                                 threshold=None):
        """图的深度优先遍历

        :param start_node:
        :param adjacency_table:
        :param depth:
        :param threshold:
        :return:
        """

        def dpt(start_node, depth):
            for neighbor in adjacency_table[start_node]:
                if not close.get(neighbor, False):
                    nodes.append(neighbor)
                    if threshold and len(nodes) >= threshold:
                        return True
                    close[neighbor] = True
                    if depth > 1:
                        dpt(neighbor, depth - 1)

        nodes = deque()
        close = {start_node: True}

        if depth:
            nodes.append(start_node)
            dpt(start_node, depth)
        else:
            open = deque()
            open.append(start_node)
            while open:
                current_node = open.pop()
                nodes.append(current_node)
                if threshold and len(nodes) >= threshold:
                    return True

                for neighbor in adjacency_table[current_node]:
                    if not close.get(neighbor, False):
                        open.append(neighbor)
                        close[neighbor] = True
        if threshold:
            return False
        return list(nodes)

    @staticmethod
    def cal_spread_size_using_gout_gin(seeds, gin_mat, gout_size: float):
        """
        gin_mat: 2d-array
        return: list
        """
        sub_mat = gin_mat[np.array(seeds)]
        virus_mat = np.cumsum(sub_mat, axis=0) > 0
        virus = virus_mat.sum(axis=1)
        sp = virus / gin_mat.shape[1] * gout_size
        return list(sp)

    @staticmethod
    def bond_percolation(adjacency_table, beta) -> list:
        sub_at = [[] for _ in range(len(adjacency_table))]
        for start_node, neighbors in enumerate(adjacency_table):
            for neighbor in neighbors:
                if random.random() <= beta:
                    sub_at[start_node].append(neighbor)
        return sub_at

    @staticmethod
    def bond_percolation_nx(adjacency_table, beta):
        """"""
        sub_g = nx.DiGraph()
        for start_node, neighbors in enumerate(adjacency_table):
            for neighbor in neighbors:
                if random.random() <= beta:
                    sub_g.add_edge(start_node, neighbor)
        return sub_g

    @staticmethod
    def gin_matrix(g, percolation_model, beta: float, times: int):
        """Get gin matrix.

        Position arguments:
        g: network
        percolation_model: 'bp' or 'nbp'
        beta: percolation rate
        times: percolation times

        Return:
        gin matrix which's shape is (g.node_num, times)
        """
        gin_matrix = sparse.dok_matrix((g.node_num, times), dtype=np.int8)
        gout_size = []
        for i in range(times):
            if percolation_model == 'bp':
                sub_at = g.bond_percolation(beta)
            elif percolation_model == 'nbp':
                sub_at = g.n_bond_percolation(beta)
            else:
                print('missing argument \'percolation_model\'')
                raise Exception
            gin, gout = Algorithm.get_gin_gout(sub_at)
            for node in gin:
                gin_matrix[node, i] = 1
            gout_size.append(len(gout))

        gin_matrix = gin_matrix.tocsr()

        return gout_size, gin_matrix

    @staticmethod
    def in_component_from_scc(rev_at, scc):
        return Algorithm.breadth_priority_traversal(scc, rev_at)

    @staticmethod
    def out_component_from_scc(adjacency_table, scc):
        return Algorithm.breadth_priority_traversal(scc, adjacency_table)

    @staticmethod
    def giant_out_component_from_adjacency_table(adjacency_table: list):
        return Algorithm.get_gout(adjacency_table)

    @staticmethod
    def get_sccs(adjacency_table):
        rev_at = Algorithm.reverse_graph(adjacency_table)
        visit_sequence = Algorithm.post_travel(rev_at)
        sccs = Algorithm.strongly_connected_components(
            adjacency_table, visit_sequence)
        return sccs

    @staticmethod
    def get_sccs_nx(nx_g):
        nx_sccs = nx.strongly_connected_components(nx_g)
        return nx_sccs

    @staticmethod
    def NewDiGraph(adjacency_table):
        nx_g = nx.DiGraph()
        for start_node, neighbors in enumerate(adjacency_table):
            for neighbor in neighbors:
                nx_g.add_edge(start_node, neighbor)
        return nx_g

    @staticmethod
    def get_gout(adjacency_table):
        rev_at = Algorithm.reverse_graph(adjacency_table)
        visit_sequence = Algorithm.post_travel(rev_at)
        sccs = Algorithm.strongly_connected_components(
            adjacency_table, visit_sequence)
        gscc = max(sccs, key=lambda scc: len(scc))
        gout = Algorithm.out_component_from_scc(adjacency_table, gscc)
        return gout

    @staticmethod
    def get_gin(adjacency_table) -> list:
        rev_at = Algorithm.reverse_graph(adjacency_table)
        visit_sequence = Algorithm.post_travel(rev_at)
        sccs = Algorithm.strongly_connected_components(
            adjacency_table, visit_sequence)
        gscc = max(sccs, key=lambda scc: len(scc))
        gin = Algorithm.in_component_from_scc(rev_at, gscc)
        return gin

    @staticmethod
    def get_gin_gout(adjacency_table):
        rev_at = Algorithm.reverse_graph(adjacency_table)
        visit_sequence = Algorithm.post_travel(rev_at)
        sccs = Algorithm.strongly_connected_components(
            adjacency_table, visit_sequence)
        gscc = max(sccs, key=lambda scc: len(scc))
        gin = Algorithm.in_component_from_scc(rev_at, gscc)
        gout = Algorithm.out_component_from_scc(adjacency_table, gscc)
        return gin, gout

    @staticmethod
    def reverse_graph(adjacency_table):
        """改变每一条边的指向

        :param adjacency_table:
        :return:
        """
        reverse_at = [[] for _ in range(len(adjacency_table))]
        for node, neighbors in enumerate(adjacency_table):
            for neighbor in neighbors:
                reverse_at[neighbor].append(node)
        return reverse_at

    @staticmethod
    def post_travel(adjacency_table):
        flags = [True] * len(adjacency_table)
        visit_sequence = []
        for start_node in range(len(adjacency_table)):
            if flags[start_node]:
                queue = deque()
                stack = deque()
                queue.append(start_node)
                flags[start_node] = False
                while queue:
                    current_node = queue.pop()
                    stack.appendleft(current_node)
                    for neighbor in adjacency_table[current_node]:
                        if flags[neighbor]:
                            queue.append(neighbor)
                            flags[neighbor] = False
                visit_sequence += list(stack)
        return visit_sequence

    @staticmethod
    def strongly_connected_components(adjacency_table, visit_sequence):
        """根据特定的节点顺序求图的所有强连通量"""
        sccs = []
        flags = [True] * len(adjacency_table)
        for i in range(len(visit_sequence)):
            start_node = visit_sequence[-i - 1]
            if flags[start_node]:
                # DFS
                scc = deque()  # strongly connected component
                stack = deque()
                stack.append(start_node)
                flags[start_node] = False
                while stack:
                    current_node = stack.pop()
                    scc.append(current_node)
                    for neib in adjacency_table[current_node]:
                        if flags[neib]:
                            stack.append(neib)
                            flags[neib] = False
                sccs.append(list(scc))
        return sccs

    @staticmethod
    def is_virus(seed, adjacency_table, threshold):

        open = deque([seed])
        close = {seed: True}
        spread_size = 1

        while open:
            current_node = open.pop()
            for neighbor in adjacency_table[current_node]:
                if neighbor not in close:
                    if spread_size + 1 >= threshold:
                        return True
                    open.appendleft(neighbor)
                    close[neighbor] = True
                    spread_size += 1

        return False

    @staticmethod
    def truncation_nsir_spread(g, beta, sources: list, threshold):
        """"""
        sub_at = [[] for _ in range(g.node_num)]
        open_table = deque(sources)
        flag = {node: node for node in open_table}  # 标记节点是属于哪颗传播树, 同时标记节点是否被感染
        volumes = {node: 1 for node in open_table}  # 记录每颗树的大小
        link = {}  # 记录是否连接到其它树
        factor = g.edge_num * beta / g.sum_edge_activity

        virus = []
        while open_table:
            current_node = open_table.pop()
            for neighbor in g.adjacency_table[current_node]:
                if neighbor not in flag:
                    if random.random() <= g.edge_activity[neighbor] * factor:
                        open_table.appendleft(neighbor)
                        flag[neighbor] = flag[current_node]
                        sub_at[current_node].append(neighbor)
                        volumes[flag[current_node]] += 1
                        if volumes[flag[current_node]] >= threshold:
                            virus.append(flag[current_node])
                else:
                    if flag[neighbor] != flag[current_node]:
                        if random.random() <= g.edge_activity[neighbor] * factor:
                            sub_at[current_node].append(neighbor)
                            link[flag[current_node]] = True
                            volumes[flag[current_node]] += 1
                            if volumes[flag[current_node]] >= threshold:
                                virus.append(flag[current_node])

        for seed in sources:
            if seed in link and seed not in virus:
                if Algorithm.is_virus(seed, sub_at, threshold):
                    virus.append(seed)

        return virus

    @staticmethod
    def truncation_sir_spread(g, beta, sources: list, threshold):
        """Simulate sir spread, and stop node's spread when it reach spread
        size of threshold.

        Return:
        nodes that reach spread size of threshold.
        """
        sub_at = [[] for _ in range(g.node_num)]
        open_table = deque(sources)
        flag = {node: node for node in open_table}  # 标记节点是属于哪颗传播树, 同时标记节点是否被感染
        volumes = {node: 1 for node in open_table}  # 记录每颗树的大小
        link = {}  # 记录是否连接到其它树

        virus = []
        while open_table:
            current_node = open_table.pop()
            for neighbor in g.adjacency_table[current_node]:
                if neighbor not in flag:
                    if random.random() <= beta:
                        open_table.appendleft(neighbor)
                        flag[neighbor] = flag[current_node]
                        sub_at[current_node].append(neighbor)
                        volumes[flag[current_node]] += 1
                        if volumes[flag[current_node]] >= threshold:
                            virus.append(flag[current_node])
                else:
                    if flag[neighbor] != flag[current_node]:
                        if random.random() <= beta:
                            sub_at[current_node].append(neighbor)
                            link[flag[current_node]] = True
                            volumes[flag[current_node]] += 1
                            if volumes[flag[current_node]] >= threshold:
                                virus.append(flag[current_node])

        for seed in sources:
            if seed in link and seed not in virus:
                if Algorithm.is_virus(seed, sub_at, threshold):
                    virus.append(seed)

        return virus

    @staticmethod
    def get_k_shell_for_undirected_graph(adjacency_table):
        """计算每个点所属的核.
        删点，直到遇到下一个shell

        :param: adjacency_table
        :return:
        """
        adjacency_table_copy = copy.deepcopy(adjacency_table)
        node_num = len(adjacency_table_copy)
        in_at = Algorithm.reverse_graph(adjacency_table_copy)

        k_shell = [0]*node_num
        remained = list(range(node_num))
        while remained:
            degrees = [len(adjacency_table_copy[node]) for node in remained]
            current_k_shell = min(degrees)
            print(current_k_shell)
            min_degree = current_k_shell
            while remained and min_degree <= current_k_shell:
                print('\t', min_degree)
                for i in range(len(remained)):
                    if degrees[i] == min_degree:
                        for neighbor in in_at[remained[i]]:
                            if adjacency_table_copy[neighbor]:
                                adjacency_table_copy[neighbor].remove(
                                    remained[i])
                        adjacency_table_copy[remained[i]].clear()
                new_remained = list()
                for node in remained:
                    if adjacency_table_copy[node]:
                        new_remained.append(node)
                    else:
                        k_shell[node] = current_k_shell
                remained = new_remained
                if remained:
                    degrees = [len(adjacency_table_copy[node])
                               for node in remained]
                    min_degree = min(degrees)
        return k_shell

    @staticmethod
    def get_betweenness(adjacency_table):
        node_num = len(adjacency_table)
        cb = [0.0] * node_num
        V = list(range(node_num))
        for s in V:
            if (s + 1) % 10 == 0:
                print(s + 1, node_num, sep='/')
            st = deque()
            P = [[] for node in V]
            sigma = [0.0] * node_num
            sigma[s] = 1
            d = [-1] * node_num
            d[s] = 0
            Q = deque()
            Q.append(s)
            while Q:
                v = Q.popleft()
                st.append(v)
                for w in adjacency_table[v]:
                    if d[w] < 0:
                        Q.append(w)
                        d[w] = d[v] + 1
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)
            delta = [0.0] * node_num
            while st:
                w = st.pop()
                for v in P[w]:
                    delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
                if w != s:
                    cb[w] += delta[w]
        return cb

    @staticmethod
    def get_closeness(adjacency_table, node):
        """CL(i) = sum([1 / d(i,j) for j != i]) / (n - 1)
        """
        node_num = len(adjacency_table)
        inverse_distance_sum = 0.0
        flag = [True] * node_num
        queue = deque([node])
        flag[node] = False
        current_level = 0
        current_level_count = 1
        while queue:
            print(current_level, current_level_count, sep=':', end=', ')
            next_level_count = 0
            for i in range(current_level_count):
                current_node = queue.popleft()
                for neighbor in adjacency_table[current_node]:
                    if flag[neighbor]:
                        queue.append(neighbor)
                        flag[neighbor] = False
                        next_level_count += 1
            if current_level:
                inverse_distance_sum += current_level_count * \
                    (1 / current_level)
            current_level += 1
            current_level_count = next_level_count
        closeness = inverse_distance_sum / (node_num - 1)
        print()
        return closeness

    @staticmethod
    def get_collective_influence(adjacency_table, node):
        # tid = multiprocessing.current_process()
        # print(tid, node, sep=':\t')
        friend1, friend2, friend3 = [], [], []
        visited = [node]
        flag = [True] * len(adjacency_table)
        flag[node] = False
        friend1 = adjacency_table[node]
        for n in friend1:
            flag[n] = False
        for n in friend1:
            for neighbor in adjacency_table[n]:
                if flag[neighbor]:
                    friend2.append(neighbor)
                    flag[neighbor] = False
        for n in friend2:
            for neighbor in adjacency_table[n]:
                if flag[neighbor]:
                    friend3.append(neighbor)
                    flag[neighbor] = False
        return len(friend1)+len(friend2)+len(friend3)


def get_gin_mat_gout_szs(g, beta: float, times: int = 1000):
    gin_mat = sparse.dok_matrix((g.node_num, times), dtype=np.int8)
    gout_szs = []
    for t in range(times):
        sub_at = g.bond_percolation(beta)
        gin, gout = Algorithm.get_gin_gout(sub_at)
        for node in gin:
            gin_mat[node, t] = 1
        gout_szs.append(len(gout))
    gin_mat = gin_mat.tocsr()
    return gin_mat, gout_szs


def get_n_gin_mat_gout_szs(g, beta: float, times: int = 1000):
    gin_mat = sparse.dok_matrix((g.node_num, times), dtype=np.int8)
    gout_szs = []
    for t in range(times):
        sub_at = g.n_bond_percolation(beta)
        gin, gout = Algorithm.get_gin_gout(sub_at)
        for node in gin:
            gin_mat[node, t] = 1
        gout_szs.append(len(gout))
    gin_mat = gin_mat.tocsr()
    return gin_mat, gout_szs


def pbga_ga(candidates, virus_set):
    while [] in virus_set:
        virus_set.remove([])
    sorted_candidates = []
    while virus_set:
        appear = {}
        for index, nodes in enumerate(virus_set):
            for node in nodes:
                if node not in appear:
                    appear[node] = []
                appear[node].append(index)
        target_candidate = max(appear.keys(), key=lambda x: len(appear[x]))
        count = 0
        for index in appear[target_candidate]:
            virus_set.pop(index - count)
            count += 1
        sorted_candidates.append(target_candidate)
    other_candidates = random.sample(list(set(
        candidates) - set(sorted_candidates)), len(candidates) - len(sorted_candidates))
    sorted_candidates += other_candidates

    return sorted_candidates


def pbga_gin_mat(candidates, gin_mat):
    virus_set = []
    for i in range(gin_mat.shape[1]):
        virus = []
        for candidate in candidates:
            if gin_mat[candidate, i]:
                virus.append(candidate)
        virus_set.append(virus)

    seeds = pbga_ga(candidates, virus_set)
    return seeds


def pbga_gin_mat_v2(candidates, gin_mat):
    c_mat = gin_mat[np.array(candidates)]
    c_mat = np.hstack((c_mat, np.array([candidates]).T))
    while c_mat.shape[1] > 1:
        v = np.argmax(c_mat[:, :-1].sum(axis=1))
        inds = np.where(c_mat[v, :-1] == 0)
        c_mat = c_mat[:, inds[0]]
        