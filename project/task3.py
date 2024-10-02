import functools
import operator
import numpy as np
import scipy as sp

from dataclasses import dataclass
from typing import Iterable

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State
from networkx import MultiDiGraph
from project.task2 import graph_to_nfa, regex_to_dfa


@dataclass
class AdjacencyMatrixFAData:
    states: set[State]
    start_states: set[State]
    final_states: set[State]
    states_indices: dict[State, int]
    boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix]


class AdjacencyMatrixFA:
    def __init__(
        self, automaton: NondeterministicFiniteAutomaton | AdjacencyMatrixFAData
    ):
        self._states: set[State] = automaton.states
        self._start_states: set[State] = automaton.start_states
        self._final_states: set[State] = automaton.final_states
        self._states_amount: int = len(self._states)

        self._states_indices: dict[State, int] = (
            automaton.states_indices
            if isinstance(automaton, AdjacencyMatrixFAData)
            else {st: i for (i, st) in enumerate(automaton.states)}
        )
        self._start_states_indices: set[int] = set(
            self._states_indices[st] for st in automaton.start_states
        )
        self._final_states_indices: set[int] = set(
            self._states_indices[st] for st in automaton.final_states
        )

        self._matrix_size: tuple[int, int] = (self._states_amount, self._states_amount)
        self._boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix] = {}

        if isinstance(automaton, AdjacencyMatrixFAData):
            self._boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix] = (
                automaton.boolean_decomposition
            )
            return
        graph = automaton.to_networkx()
        for u, v, label in graph.edges(data="label"):
            if label:
                symbol = Symbol(label)
                if symbol not in self._boolean_decomposition:
                    self._boolean_decomposition[symbol] = sp.sparse.csc_matrix(
                        self._matrix_size, dtype=bool
                    )
                self._boolean_decomposition[symbol][
                    self._states_indices[u], self._states_indices[v]
                ] = True

    @property
    def boolean_decomposition(self) -> dict[Symbol, sp.sparse.csc_matrix]:
        return self._boolean_decomposition

    @property
    def states(self) -> set[State]:
        return self._states

    @property
    def start_states(self) -> set[State]:
        return self._start_states

    @property
    def final_states(self) -> set[State]:
        return self._final_states

    @property
    def states_amount(self) -> int:
        return self._states_amount

    @property
    def states_indices(self) -> dict[State, int]:
        return self._states_indices

    @property
    def start_states_indices(self) -> set[int]:
        return self._start_states_indices

    @property
    def final_states_indices(self) -> set[int]:
        return self._final_states_indices

    @property
    def start_configuration(self) -> np.ndarray:
        start_config = np.zeros(self._states_amount, dtype=bool)
        for start_state_index in self._start_states_indices:
            start_config[start_state_index] = True
        return start_config

    @property
    def final_configuration(self) -> np.ndarray:
        final_config = np.zeros(self._states_amount, dtype=bool)
        for final_state_index in self._final_states_indices:
            final_config[final_state_index] = True
        return final_config

    def transitive_closure(self) -> sp.sparse.csc_matrix:
        matrices_list = list(self._boolean_decomposition.values())
        init_matrix = sp.sparse.csc_matrix(self._matrix_size, dtype=bool)
        main_diagonal_indices = np.arange(self._matrix_size[0])
        init_matrix[main_diagonal_indices, main_diagonal_indices] = True
        common_matrix = functools.reduce(operator.add, matrices_list, init_matrix)
        closure = common_matrix**self._states_amount
        return closure

    def accepts(self, word: Iterable[Symbol]) -> bool:
        start_config = self.start_configuration
        final_config = self.final_configuration

        current_config = start_config.copy()

        for symbol in word:
            if symbol not in self._boolean_decomposition:
                return False
            matrix = self._boolean_decomposition[symbol]
            current_config = current_config @ matrix.toarray()
        return np.any(current_config & final_config)

    def is_empty(self) -> bool:
        transitive_closure = self.transitive_closure()
        empty = True
        for start in self._start_states_indices:
            for final in self._final_states_indices:
                empty = False if transitive_closure[start, final] else empty
        return empty


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    states: set[State] = set()
    start_states: set[State] = set()
    final_states: set[State] = set()
    states_indices: dict[State, int] = {}
    boolean_decomposition: dict[Symbol, sp.sparse.csc_matrix] = {}

    for first_automaton_state in automaton1.states:
        for second_automaton_state in automaton2.states:
            first_automaton_state_index = automaton1.states_indices[
                first_automaton_state
            ]
            second_automaton_state_index = automaton2.states_indices[
                second_automaton_state
            ]

            new_state = State((first_automaton_state, second_automaton_state))
            new_automaton_state_index = (
                automaton2.states_amount * first_automaton_state_index
                + second_automaton_state_index
            )

            states.add(new_state)
            states_indices[new_state] = new_automaton_state_index

            if (
                first_automaton_state in automaton1.start_states
                and second_automaton_state in automaton2.start_states
            ):
                start_states.add(new_state)
            if (
                first_automaton_state in automaton1.final_states
                and second_automaton_state in automaton2.final_states
            ):
                final_states.add(new_state)

    for symbol in automaton1.boolean_decomposition:
        if symbol not in automaton2.boolean_decomposition:
            continue

        first_automaton_symbol_adj = automaton1.boolean_decomposition[symbol]
        second_automaton_symbol_adj = automaton2.boolean_decomposition[symbol]

        new_automaton_symbol_adj = sp.sparse.kron(
            first_automaton_symbol_adj, second_automaton_symbol_adj, format="csc"
        )
        boolean_decomposition[symbol] = new_automaton_symbol_adj

    return AdjacencyMatrixFA(
        AdjacencyMatrixFAData(
            states=states,
            start_states=start_states,
            final_states=final_states,
            states_indices=states_indices,
            boolean_decomposition=boolean_decomposition,
        )
    )


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    regex_dfa = regex_to_dfa(regex)

    graph_nfa_adj_matrix = AdjacencyMatrixFA(graph_nfa)
    regex_dfa_adj_matrix = AdjacencyMatrixFA(regex_dfa)

    adj_matrix_intersection = intersect_automata(
        graph_nfa_adj_matrix, regex_dfa_adj_matrix
    )
    reachability_matrix = adj_matrix_intersection.transitive_closure()

    result = set()

    inverted_states_indices_dict = {
        value: key for key, value in adj_matrix_intersection.states_indices.items()
    }

    for start in adj_matrix_intersection.start_states_indices:
        for final in adj_matrix_intersection.final_states_indices:
            if reachability_matrix[start, final]:
                graph_start_state = inverted_states_indices_dict[start].value[0]
                graph_final_state = inverted_states_indices_dict[final].value[0]
                result.add((graph_start_state, graph_final_state))

    return result
