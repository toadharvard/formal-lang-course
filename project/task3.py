from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton as NFA, Symbol
from numpy.typing import NDArray
import numpy as np
from numpy import bool_
from scipy.sparse import csr_array, kron

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, fa: NFA | None):
        if fa is None:
            self.start_states: set[int] = set()
            self.final_states: set[int] = set()
            self.states: dict[Any, int] = {}
            self.boolean_decomposition: dict[Symbol, csr_array] = {}
            return

        graph = fa.to_networkx()
        self.states: dict[Any, int] = {st: i for (i, st) in enumerate(graph.nodes)}
        number_of_states = len(self.states)
        self.start_states = set(self.states[st] for st in fa.start_states)
        self.final_states = set(self.states[st] for st in fa.final_states)

        transitions = defaultdict(
            lambda: np.zeros(
                (number_of_states, number_of_states),
                dtype=bool_,
            )
        )

        for st1, st2, label in graph.edges(data="label"):
            if not label:
                continue
            sym = Symbol(label)
            transitions[sym][self.states[st1], self.states[st2]] = True

        self.boolean_decomposition = {
            sym: csr_array(matrix) for (sym, matrix) in transitions.items()
        }

    def accepts(self, word: Iterable[Symbol]) -> bool:
        @dataclass(frozen=True)
        class Config:
            state: int
            word: list[Symbol]

        word = list(word)
        number_of_states = len(self.states)

        stack = [Config(st, word) for st in self.start_states]
        while stack:
            cfg = stack.pop()
            if not cfg.word:
                if cfg.state in self.final_states:
                    return True
                continue

            sym = cfg.word[0]
            adj = self.boolean_decomposition[sym]
            if adj is None:
                continue

            for next_state in range(number_of_states):
                if adj[cfg.state, next_state]:
                    stack.append(Config(next_state, cfg.word[1:]))
        return False

    def transitive_closure(self) -> NDArray[bool_]:
        number_of_states = len(self.states)
        if not self.boolean_decomposition:
            return np.eye(number_of_states, dtype=bool_)

        combined: csr_array = sum(self.boolean_decomposition.values())
        combined.setdiag(True)
        tc = np.linalg.matrix_power(combined.toarray(), number_of_states)
        return tc

    def is_empty(self) -> bool:
        tc = self.transitive_closure()
        return not any(
            tc[start, final]
            for start, final in product(self.start_states, self.final_states)
        )


def intersect_automata(
    fa1: AdjacencyMatrixFA, fa2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    result = AdjacencyMatrixFA(None)

    for st1, st2 in product(fa1.states, fa2.states):
        idx1, idx2 = fa1.states[st1], fa2.states[st2]
        result_idx = len(fa2.states) * idx1 + idx2

        if idx1 in fa1.start_states and idx2 in fa2.start_states:
            result.start_states.add(result_idx)
        if idx1 in fa1.final_states and idx2 in fa2.final_states:
            result.final_states.add(result_idx)

        result.states[(st1, st2)] = result_idx

    for sym, adj1 in fa1.boolean_decomposition.items():
        if sym not in fa2.boolean_decomposition:
            continue

        adj2 = fa2.boolean_decomposition[sym]
        result.boolean_decomposition[sym] = kron(adj1, adj2, format="csr")

    return result


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    all_nodes = {int(n) for n in graph.nodes}
    start_nodes = start_nodes or all_nodes
    final_nodes = final_nodes or all_nodes

    graph_mfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_dfa = regex_to_dfa(regex)
    regex_mfa = AdjacencyMatrixFA(regex_dfa)

    inter_mfa = intersect_automata(graph_mfa, regex_mfa)
    inter_tc = inter_mfa.transitive_closure()

    return {
        (start, final)
        for start, final in product(start_nodes, final_nodes)
        for regex_start, regex_final in product(
            regex_dfa.start_states, regex_dfa.final_states
        )
        if inter_tc[
            inter_mfa.states[(start, regex_start)],
            inter_mfa.states[(final, regex_final)],
        ]
    }
