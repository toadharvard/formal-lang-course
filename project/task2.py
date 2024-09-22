from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    EpsilonNFA,
    NondeterministicFiniteAutomaton as NFA,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    epsilon_nfa: EpsilonNFA = Regex(regex).to_epsilon_nfa()
    dfa: DeterministicFiniteAutomaton = epsilon_nfa.to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NFA:
    nfa: NFA = NFA.from_networkx(graph).remove_epsilon_transitions()

    all_states = set(int(node) for node in graph.nodes)

    actual_start_states = start_states or all_states
    actual_final_states = final_states or all_states

    for state in actual_start_states:
        nfa.add_start_state(state)
    for state in actual_final_states:
        nfa.add_final_state(state)
    return nfa
