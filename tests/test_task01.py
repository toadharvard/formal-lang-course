from pathlib import Path
import project.graph_utils as utils


def test_travel_statistics():
    expected = utils.GraphStats(
        number_of_nodes=131,
        number_of_edges=277,
        set_of_lables=frozenset(
            {
                "range",
                "domain",
                "complementOf",
                "equivalentClass",
                "inverseOf",
                "comment",
                "minCardinality",
                "type",
                "versionInfo",
                "rest",
                "oneOf",
                "hasPart",
                "someValuesFrom",
                "hasAccommodation",
                "differentFrom",
                "subClassOf",
                "disjointWith",
                "first",
                "unionOf",
                "hasValue",
                "intersectionOf",
                "onProperty",
            }
        ),
    )
    graph = utils.load_graph_by_name("travel")
    actual = utils.get_statistics(graph)
    assert expected == actual


def test_bzip_statistics():
    expected = utils.GraphStats(
        number_of_nodes=632, number_of_edges=556, set_of_lables=frozenset({"a", "d"})
    )
    graph = utils.load_graph_by_name("bzip")
    actual = utils.get_statistics(graph)
    assert expected == actual


def test_saving_labeled_two_cycles_graph_as_dot():
    actual = Path("./tests/resources/actual_two_cycles_graph.dot")
    expected = Path("./tests/resources/expected_two_cycles_graph.dot")

    graph = utils.make_labeled_two_cycles_graph(5, 5, ("a", "b"))
    utils.save_graph_as_dot(graph, actual)

    with open(expected, "r") as expected_file:
        with open(actual, "r") as actual_file:
            assert expected_file.read() == actual_file.read()
