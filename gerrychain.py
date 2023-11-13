import networkx as nx
from networkx.algorithms import tree

import random
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout


def predecessors(h, root):
    return {a: b for a, b in nx.bfs_predecessors(h, root)}


def random_spanning_tree(graph):
    for edge in graph.edges:
        graph.edges[edge]["weight"] = random.random()

    spanning_tree = tree.maximum_spanning_tree(
        graph, algorithm="kruskal", weight="weight"
    )
    return spanning_tree


class PopulatedGraph:
    def __init__(self, graph, populations, ideal_pop, epsilon):
        self.graph = graph
        self.subsets = {node: {node} for node in graph}
        self.population = populations.copy()
        self.ideal_pop = ideal_pop
        self.epsilon = epsilon
        self._degrees = {node: graph.degree(node) for node in graph}

    def __iter__(self):
        return iter(self.graph)

    def degree(self, node):
        return self._degrees[node]

    def contract_node(self, node, parent):
        self.population[parent] += self.population[node]
        self.subsets[parent] |= self.subsets[node]
        self._degrees[parent] -= 1

    def has_ideal_population(self, node):
        return (
            abs(self.population[node] - self.ideal_pop) < self.epsilon * self.ideal_pop
        )


class PopulatedGraph_using_bounds:
    def __init__(self, graph, populations, p_minimum, p_maximum):
        self.graph = graph
        self.subsets = {node: {node} for node in graph}
        self.population = populations.copy()
        self.p_minimum = p_minimum
        self.p_maximum = p_maximum
        self._degrees = {node: graph.degree(node) for node in graph}

    def __iter__(self):
        return iter(self.graph)

    def degree(self, node):
        return self._degrees[node]

    def contract_node(self, node, parent):
        self.population[parent] += self.population[node]
        self.subsets[parent] |= self.subsets[node]
        self._degrees[parent] -= 1

    def has_ideal_population(self, node):
        return (
            self.population[node] > self.p_minimum and self.population[node] < self.p_maximum
        )


def contract_leaves_until_balanced_or_none(h):
    # this used to be greater than 2 but failed on small grids:(
    root = random.choice([x for x in h if h.degree(x) > 1])
    # BFS predecessors for iteratively contracting leaves
    pred = predecessors(h.graph, root)

    leaves = deque(x for x in h if h.degree(x) == 1)
    while len(leaves) > 0:
        leaf = leaves.popleft()
        if h.has_ideal_population(leaf):
            return h.subsets[leaf]
        # Contract the leaf:
        # print("leaf is",leaf)
        # print("pred is",pred)
        # if leaf in pred:
        parent = pred[leaf]
        h.contract_node(leaf, parent)
        if h.degree(parent) == 1 and parent != root:
            leaves.append(parent)
    return None


Cut = namedtuple("Cut", "edge subset")


def find_balanced_edge_cuts(h):
    # this used to be greater than 2 but failed on small grids:(
    root = random.choice([x for x in h if h.degree(x) > 1])
    # BFS predecessors for iteratively contracting leaves
    pred = predecessors(h.graph, root)

    cuts = []
    leaves = deque(x for x in h if h.degree(x) == 1)
    while len(leaves) > 0:
        leaf = leaves.popleft()
        if h.has_ideal_population(leaf):
            cuts.append(Cut(edge=(leaf, pred[leaf]), subset=h.subsets[leaf].copy()))
        # Contract the leaf:
        parent = pred[leaf]
        h.contract_node(leaf, parent)
        if h.degree(parent) == 1 and parent != root:
            leaves.append(parent)
    return cuts


def bipartition_tree( graph,pop_col,    pop_target,epsilon,node_repeats,spanning_tree=None):
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    balanced_subtree = None
    if spanning_tree is None:
        spanning_tree = random_spanning_tree(graph)
    restarts = 0

    while balanced_subtree is None:
        if restarts == node_repeats:
            spanning_tree = random_spanning_tree(graph)
            restarts = 0
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        balanced_subtree = contract_leaves_until_balanced_or_none(h)
        restarts += 1

    color_map = []
    for node in spanning_tree:
        if node in balanced_subtree:
            color_map.append('red')
        else:
            color_map.append('blue')
        spanning_tree.nodes[node]['pos'] = (graph.nodes[node]['x'],graph.nodes[node]['y'])

    # print("plotting")
    # # pos = graphviz_layout(spanning_tree)
    # pos = nx.get_node_attributes(spanning_tree,'pos')
    # nx.draw(spanning_tree, pos, node_size=5, node_color=color_map)
    plt.show()

    return balanced_subtree


def bipartition_tree_using_bounds(graph,pop_col,P_minimum,P_maximum,node_repeats,spanning_tree=None):

    populations = {node: graph.nodes[node][pop_col] for node in graph}

    if spanning_tree is None:
        spanning_tree = random_spanning_tree(graph)
    restarts = 0

    balanced_subtree = None
    while balanced_subtree is None:
        if restarts == node_repeats:
            spanning_tree = random_spanning_tree(graph)
            restarts = 0
        # h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        h = PopulatedGraph_using_bounds(spanning_tree, populations, P_minimum, P_maximum) # prepares a data structure useful for contration in the next step
        balanced_subtree = contract_leaves_until_balanced_or_none(h)
        restarts += 1

    color_map = []
    for node in spanning_tree:
        if node in balanced_subtree:
            color_map.append('red')
        else:
            color_map.append('blue')
        spanning_tree.nodes[node]['pos'] = (graph.nodes[node]['x'],graph.nodes[node]['y'])

    # print("plotting")
    # # pos = graphviz_layout(spanning_tree)
    # pos = nx.get_node_attributes(spanning_tree,'pos')
    # nx.draw(spanning_tree, pos, node_size=5, node_color=color_map)
    # plt.show()

    return balanced_subtree


def bipartition_tree_random(graph,pop_col,pop_target,epsilon,node_repeats=1,spanning_tree=None,choice = random.choice):
    populations = {node: graph.nodes[node][pop_col] for node in graph}

    possible_cuts = []
    if spanning_tree is None:
        spanning_tree = random_spanning_tree(graph)

    while len(possible_cuts) == 0:
        spanning_tree = random_spanning_tree(graph)
        h = PopulatedGraph(spanning_tree, populations, pop_target, epsilon)
        possible_cuts = find_balanced_edge_cuts(h)

    return choice(possible_cuts).subset


def recursive_tree_part(
    graph, parts, pop_target, pop_col, epsilon, node_repeats=1, method=bipartition_tree
):
    """Uses :func:`~gerrychain.tree.bipartition_tree` recursively to partition a tree into
    ``len(parts)`` parts of population ``pop_target`` (within ``epsilon``). Can be used to
    generate initial seed plans or to implement ReCom-like "merge walk" proposals.

    :param graph: The graph
    :param parts: Iterable of part labels (like ``[0,1,2]`` or ``range(4)``
    :param pop_target: Target population for each part of the partition
    :param pop_col: Node attribute key holding population data
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts
        of the partition can be
    :param node_repeats: Parameter for :func:`~gerrychain.tree_methods.bipartition_tree` to use.
    :return: New assignments for the nodes of ``graph``.
    :rtype: dict
    """
    flips = {}
    remaining_nodes = set(graph.nodes)
    # We keep a running tally of deviation from ``epsilon`` at each partition
    # and use it to tighten the population constraints on a per-partition
    # basis such that every partition, including the last partition, has a
    # population within +/-``epsilon`` of the target population.
    # For instance, if district n's population exceeds the target by 2%
    # with a +/-2% epsilon, then district n+1's population should be between
    # 98% of the target population and the target population.
    debt = 0

    for part in parts[:-1]:
        min_pop = max(pop_target * (1 - epsilon), pop_target * (1 - epsilon) - debt)
        max_pop = min(pop_target * (1 + epsilon), pop_target * (1 + epsilon) - debt)
        nodes = method(
            graph.subgraph(remaining_nodes),
            pop_col=pop_col,
            pop_target=(min_pop + max_pop) / 2,
            epsilon=(max_pop - min_pop) / (2 * pop_target),
            node_repeats=node_repeats,
        )

        part_pop = 0
        for node in nodes:
            flips[node] = part
            part_pop += graph.nodes[node][pop_col]
        debt += part_pop - pop_target
        remaining_nodes -= nodes

    # All of the remaining nodes go in the last part
    for node in remaining_nodes:
        flips[node] = parts[-1]

    return flips
