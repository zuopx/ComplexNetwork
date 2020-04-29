import copy
import random
from collections import deque

from .algorithm import Algorithm


class Graph:
    """图的文件描述和拓扑信息

    """
    def __init__(self, name, adjacency_table, description=None):
        self.name = name
        self.description = description
        self.adjacency_table = copy.deepcopy(adjacency_table)
        self.node_num = len(adjacency_table)

    def sir_spread(self, sources: list, beta: float):
        """Call Algorithm.sir_spread method."""
        return Algorithm.sir_spread(self.adjacency_table, sources, beta)


class UndirectedGraph(Graph):
    """无向图

    """
    def __init__(self, name, adjacency_table, description=None):
        super().__init__(name, adjacency_table, description)
        self.degree = [len(neighbors) for neighbors in adjacency_table]
        self.edge_num = sum(self.degree)


class DirectedGraph(Graph):
    """Directed graph."""
    def __init__(self, name: str, adjacency_table, alpha=0.68, description=None):
        super().__init__(name, adjacency_table, description)
        self.out_degree = [len(neighbors) for neighbors in adjacency_table]
        self.in_degree = [0] * self.node_num
        for neighbors in adjacency_table:
            for neighbor in neighbors:
                self.in_degree[neighbor] += 1
        self.edge_num = sum(self.out_degree)
        self.alpha = alpha
        self.node_activity = [k**self.alpha for k in self.out_degree]
        self.edge_activity = []  # self.edge_activity[i]表示以点i为终点的边的活跃度
        for a, b in zip(self.node_activity, self.in_degree):
            if b:
                self.edge_activity.append(a / b)
            else:
                self.edge_activity.append(a)
        self.sum_node_activity = sum(self.node_activity)  # 所有点的活跃度之和
        self.sum_edge_activity = sum(  # 所有边的活跃度之和，其值等于所有入度不为0的点的活跃度之和
            [
                self.node_activity[node] for node in range(self.node_num)
                if self.in_degree[node]
            ])

    def is_virus_spreading(self, seeds, beta, threshold):
        """坚定模拟广度优先传播

        :param threshold:
        :param beta:
        :param seeds:
        :return: boolean
        """
        sub_at = [[] for _ in range(self.node_num)]

        open = deque(seeds)
        flag = {node: node for node in open}  # 标记节点是属于哪颗传播树, 同时标记节点是否被感染
        volumes = {node: 1 for node in open}  # 记录每颗树的大小
        link = {}  # 记录是否连接到其它树
        factor = self.edge_num * beta / self.sum_edge_activity

        while open:
            current_node = open.pop()
            for neighbor in self.adjacency_table[current_node]:
                if neighbor not in flag:
                    if random.random() <= self.edge_activity[neighbor] *\
                            factor:
                        volumes[flag[current_node]] += 1
                        if volumes[flag[current_node]] >= threshold:
                            return True
                        open.appendleft(neighbor)
                        flag[neighbor] = flag[current_node]
                        sub_at[current_node].append(neighbor)
                else:
                    if flag[neighbor] != flag[current_node]:
                        if random.random() <= self.edge_activity[neighbor] *\
                           factor:
                            volumes[flag[current_node]] += 1
                            if volumes[flag[current_node]] >= threshold:
                                return True
                            sub_at[current_node].append(neighbor)
                            link[flag[current_node]] = True

        for seed in seeds:
            if seed in link:
                if Algorithm.is_virus(seed, sub_at, threshold):
                    return True

        return False

    def nsir_spread(self, sources: list, beta: float):
        """Spread message by non-uniform SIR model."""
        open_table = deque(sources)
        close_table = {node: True for node in open_table}
        factor = self.edge_num * beta / self.sum_edge_activity

        while open_table:
            current_node = open_table.pop()
            for neighbor in self.adjacency_table[current_node]:
                if neighbor not in close_table:
                    if random.random() <= self.edge_activity[neighbor] *\
                            factor:
                        open_table.appendleft(neighbor)
                        close_table[neighbor] = True
        infected = list(close_table)
        return infected

    def n_bond_percolation(self, beta):
        """Non-uniformly bond percolate.

        Return:
        sub_at -- adjacency table after percolation
        """
        factor = self.edge_num * beta / self.sum_edge_activity
        sub_at = [[] for _ in range(self.node_num)]
        for start_node, neighbors in enumerate(self.adjacency_table):
            for neighbor in neighbors:
                if random.random() <= self.edge_activity[neighbor] * factor:
                    sub_at[start_node].append(neighbor)
        return sub_at

    def bond_percolation(self, beta):
        return Algorithm.bond_percolation(self.adjacency_table, beta)

    def celf(self, candidates, beta, seed_num=10):
        """高效的贪婪算法

        :param candidates:
        :param beta:
        :param seed_num: 选num个点
        :return:
        """
        seeds = []
        marginal_sps = []
        times = 200

        # sp_count = 0

        def spread_(seeds, beta, times):
            # nonlocal sp_count
            # sp_count += 1
            # print('\t\t', sp_count)
            tmp_sum = 0.0
            for _ in range(times):
                tmp_sum += len(self.spread_of_candidates(seeds, beta))
            return tmp_sum / times

        for c in candidates:
            marginal_sps.append([c, spread_([c], beta, times)])
        origin_sp = 0.0
        for _ in range(seed_num):
            # print('\t', i, '/', seed_num)
            marginal_sps.sort(key=lambda x: x[1], reverse=True)
            if seeds:
                msp = spread_(seeds[:] + [marginal_sps[0][0]], beta,
                              times) - origin_sp
                marginal_sps[0][1] = msp
                count = 1 * len(marginal_sps)
                while msp < marginal_sps[1][1] and count:
                    marginal_sps.sort(key=lambda x: x[1], reverse=True)
                    msp = spread_(
                        seeds[:] + [marginal_sps[0][0]], beta, times) -\
                        origin_sp
                    marginal_sps[0][1] = msp
                    count -= 1
                origin_sp += marginal_sps[0][1]
                seeds.append(marginal_sps.pop(0)[0])
            else:
                origin_sp += marginal_sps[0][1]
                seeds.append(marginal_sps.pop(0)[0])
        return seeds

    def sccs(self):
        reverse_adjacency_table = Algorithm.reverse_graph(self.adjacency_table)
        visit_sequence = Algorithm.post_travel(reverse_adjacency_table)
        sccs = Algorithm.strongly_connected_components(self.adjacency_table,
                                                       visit_sequence)

    def giant_in_component(self):
        reverse_adjacency_table = Algorithm.reverse_graph(self.adjacency_table)
        visit_sequence = Algorithm.post_travel(reverse_adjacency_table)
        sccs = Algorithm.strongly_connected_components(self.adjacency_table,
                                                       visit_sequence)
        gscc = max(sccs, key=lambda x: len(x))
        gin = Algorithm.in_component_from_scc(reverse_adjacency_table, gscc)
        return gin

        reverse_adjacency_table = Algorithm.reverse_graph(self.adjacency_table)
        visit_sequence = Algorithm.post_travel(reverse_adjacency_table)
        sccs = Algorithm.strongly_connected_components(self.adjacency_table,
                                                       visit_sequence)
        gscc = max(sccs, key=lambda x: len(x))
        gout = Algorithm.out_component_from_scc(self.adjacency_table, gscc)
        return gout

    def giant_connected_in_out_component(self):
        reverse_adjacency_table = Algorithm.reverse_graph(self.adjacency_table)
        visit_sequence = Algorithm.post_travel(reverse_adjacency_table)
        sccs = Algorithm.strongly_connected_components(self.adjacency_table,
                                                       visit_sequence)
        gscc = max(sccs, key=lambda x: len(x))
        gin = Algorithm.in_component_from_scc(reverse_adjacency_table, gscc)
        gout = Algorithm.out_component_from_scc(self.adjacency_table, gscc)
        return gscc, gin, gout
