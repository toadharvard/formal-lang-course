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
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()

        if fa is None:
            self.states: dict[Any, int] = {}
            self.sym_to_adj_matrix: dict[Symbol, csr_array] = {}
            return

        graph = fa.to_networkx()

        self.states: dict[Any, int] = {st: i for (i, st) in enumerate(graph.nodes)}
        transitions: dict[Symbol, NDArray[bool_]] = defaultdict(
            lambda: np.zeros(
                (len(self.states), len(self.states)),
                dtype=bool_,
            )
        )

        # fill start and final states
        for state, data in graph.nodes(data=True):
            if data.get("is_start"):
                self.start_states.add(self.states[state])
            if data.get("is_final"):
                self.final_states.add(self.states[state])

        # build transition matrix for each symbol
        for idx1, idx2, sym in (
            (self.states[st1], self.states[st2], Symbol(label))
            for st1, st2, label in graph.edges(data="label")
            if label
        ):
            transitions[sym][idx1, idx2] = True

        self.sym_to_adj_matrix: dict[Symbol, csr_array] = {
            sym: csr_array(matrix) for (sym, matrix) in transitions.items()
        }

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """
        Checks if given word is accepted by this automaton.

        :param word: Symbol iterable to check
        :return: True if word is accepted, False otherwise
        """
        word = list(word)

        @dataclass(frozen=True)
        class Config:
            state: int
            word: list[Symbol]

        stack = [Config(st, word) for st in self.start_states]
        while stack:
            cfg = stack.pop()
            if not cfg.word:
                if cfg.state in self.final_states:
                    return True
                continue

            sym = cfg.word[0]
            adj = self.sym_to_adj_matrix[sym]
            if adj is None:
                continue

            for next_state in range(len(self.states)):
                if adj[cfg.state, next_state]:
                    stack.append(Config(next_state, cfg.word[1:]))

        return False

    def transitive_closure(self) -> NDArray[bool_]:
        if not self.sym_to_adj_matrix:
            return np.diag(np.ones(len(self.states), dtype=bool_))

        combined = sum(self.sym_to_adj_matrix.values())
        combined.setdiag(True)
        tc = combined.toarray()
        for pow in range(2, len(self.states) + 1):
            prev = tc
            tc = np.linalg.matrix_power(tc, pow)
            if np.array_equal(prev, tc):
                break
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

    for sym, adj1 in fa1.sym_to_adj_matrix.items():
        adj2 = fa2.sym_to_adj_matrix.get(sym)
        if adj2 is None:
            continue

        result.sym_to_adj_matrix[sym] = kron(adj1, adj2, format="csr")

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
