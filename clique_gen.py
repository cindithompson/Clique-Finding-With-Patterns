import itertools


def gen_cliques(pattern):
    """
    Find all cliques that the given pattern expands to.
    :param pattern: a Clique pattern (type CPattern)
    :return: a list of all cliques, as tuples
    """
    core = pattern.core()
    gens = pattern.gen()
    #find all combinations of the elements in the sets in gens
    core = tuple(core)
    result = []
    if not gens:
        result.append(core)
        return result
    for e in itertools.product(*gens):
        clique = core+e
        result.append(clique)
    return result


def core_results_to_cliques(core_alg_results):
    trivial = core_alg_results[0]
    #convert them to strings to be compatible w/ networkx
    results = []
    for c in trivial:
        results.append([str(x) for x in c])
    patts = core_alg_results[1]
    for p in patts:
        non_trivial = gen_cliques(p)
        for c in non_trivial:
            results.append([str(x) for x in c])
    return results

