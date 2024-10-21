import networkx as nx

from pyformlang.cfg import CFG, Variable, Production, Epsilon, Terminal


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    normal_form_cfg = cfg.to_normal_form()
    nullable = cfg.get_nullable_symbols()

    new_productions = set(normal_form_cfg.productions)
    for var in nullable:
        new_productions.add(Production(Variable(var.value), [Epsilon()]))

    weak_normal_form_cfg = CFG(
        start_symbol=cfg.start_symbol, productions=new_productions
    )
    weak_normal_form_cfg = weak_normal_form_cfg.remove_useless_symbols()

    return weak_normal_form_cfg


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    weak_normal_form = cfg_to_weak_normal_form(cfg)

    cfpq_results = set()

    for u, v, data in graph.edges(data=True):
        if (label := data.get("label")) is None:
            continue
        for production in weak_normal_form.productions:
            if (
                len(production.body) == 1
                and isinstance(production.body[0], Terminal)
                and production.body[0].value == label
            ):
                cfpq_results.add((u, production.head, v))

    nullable = weak_normal_form.get_nullable_symbols()
    for node in graph.nodes:
        for var in nullable:
            cfpq_results.add((node, var, node))

    new_results_found = True
    while new_results_found:
        new_results_found = False
        new_results = set()

        for n11, headvar1, n12 in cfpq_results:
            for n21, headvar2, n22 in cfpq_results:
                if n12 != n21:
                    continue
                for production in weak_normal_form.productions:
                    if (
                        len(production.body) == 2
                        and production.body[0] == headvar1
                        and production.body[1] == headvar2
                        and (new_triple := (n11, production.head, n22))
                        not in cfpq_results
                    ):
                        new_results.add(new_triple)
                        new_results_found = True

        cfpq_results.update(new_results)

    result_pairs = set()
    for u, var, v in cfpq_results:
        if var != weak_normal_form.start_symbol:
            continue
        if (not start_nodes or u in start_nodes) and (
            not final_nodes or v in final_nodes
        ):
            result_pairs.add((u, v))

    return result_pairs
