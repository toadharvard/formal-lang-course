import networkx as nx
import cfpq_data as cd
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphStats:
    number_of_nodes: int
    number_of_edges: int
    set_of_lables: frozenset[str]


def load_graph_by_name(name: str) -> nx.MultiDiGraph:
    graph_path = cd.download(name)
    graph = cd.graph_from_csv(graph_path)
    return graph


def get_statistics(graph: nx.MultiDiGraph) -> GraphStats:
    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()
    set_of_lables = frozenset(cd.get_sorted_labels(graph))
    return GraphStats(number_of_nodes, number_of_edges, set_of_lables)


def make_labeled_two_cycles_graph(
    n: int, m: int, labels: tuple[str, str]
) -> nx.MultiDiGraph:
    return cd.labeled_two_cycles_graph(n, m, labels=labels)


def save_graph_as_dot(graph: nx.MultiDiGraph, path: Path):
    pdg = nx.drawing.nx_pydot.to_pydot(graph)
    pdg.write_raw(path)
