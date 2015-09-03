import networkx as nx
import core_alg as ca
import clique_gen as cg
import graph_manip
from timer import Timer
import sys
import os


def compare_maximal(g):
    """
    Given the graph g, run the networkX version of finding all maximal cliques,
    and our version, and compare the sets. Fail if any are different.
    :param g: graph in networkx format
    :return: success flag
    """
    nwx_results = nx_answer(g)
    print(len(nwx_results), " expected maximal cliques for ", str(g))
    print("correct: ", nwx_results)

    our_results = [set(x) for x in cg.core_results_to_cliques( ca.process_from_g(g) )]
    our_results = rem_subsets(our_results)

    print("ours: ", our_results)
    if not same_results(nwx_results, our_results):
        return False
    return True


def core_results_to_cliques(r):
    """
    Given a list of clique patterns, return all cliques represented.
    :param r:
    :return:
    """
    return cg.core_results_to_cliques(r)


def rem_subsets(l):
    """
    Remove all sets that are subsets of another in given list.
    :param l:
    :return: new list with no subsets.
    """
    result = []
    ignore = []
    for i, ele in enumerate(l):
        if ele in ignore:
            continue
        keep = True
        for other in l[i+1:]:
            if set(ele).issubset(set(other)):
                keep = False
            elif set(other).issubset(set(ele)):
                ignore.append(other)
        if keep:
            result.append(ele)
    return result


def nx_answer(g):
    return [set(x) for x in list(nx.algorithms.clique.find_cliques(g))]


def nx_answer_from_file(filename):
    g = graph_manip.create_networkx_graph(filename)
    return nx_answer(g)


def same_results(nwx, ours):
    success = True
    for c in nwx:
        if c in ours:
            ours.remove(c)
        else:
            print("   got different results, following not in ours:", str(c))
            success = False
    if ours:
        print("  got different results, following not in theirs:", str(ours))
        success = False
    return success


def test_folder(folder, perms=False):
    """
    Test if all DIMACS formatted graphs in a file are correctly analyzed by DESC,
    as compared to standard clique finding algorithm.
    :param folder:
    :param perms: Flag indicating whether to test all permutations of the graphs.
    :return: Success indicator.
    """
    files = os.listdir(folder)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    success = True
    for f in files:
        full_name = folder + f
        if perms:
            if not test_permutations(full_name):
                print(f, " failed")
                success = False
            else:
                print(f, " succeeded")
        else:
            g = graph_manip.create_networkx_graph(full_name)
            if not compare_maximal(g):
                print(f, " failed")
                success = False
            else:
                print(f, " succeeded")
    return success


def test_permutations(filename):
    adj = ca.process_graph_file(filename)
    adj_perms = graph_manip.all_permutations(adj)
    print(filename, len(adj_perms))
    if len(adj_perms) > 10000:
        print("more than 10K perms! Just test basic", len(adj_perms), filename)
        g = graph_manip.create_networkx_graph(filename)
        if not compare_maximal(g):
            print(filename, " failed")
            return False
        return True
    for p in adj_perms:
        this_g = graph_manip.convert_to_nx(p)
        nwx_results = [set(x) for x in list(nx.algorithms.clique.find_cliques(this_g))]

        this_res = ca.processing_steps(p)
        our_results = [set(x) for x in cg.core_results_to_cliques(this_res)]
        our_results = rem_subsets(our_results)
        if not same_results(nwx_results, our_results):
            print("adj failed", p)
            print(graph_manip.pretty_print(p))
            return False
    return True


def time_compare(filename):
    g = graph_manip.create_networkx_graph(filename)
    with Timer() as t:
        nx.algorithms.clique.find_cliques(g)
    print("nx elapsed: %s", t.secs)
    with Timer() as t:
        ca.process_from_g(g)
    print("DESC elapsed: %s", t.secs)


def time_ours(filename):
    g = graph_manip.create_networkx_graph(filename)
    with Timer() as t:
        res = ca.process_from_g(g)
    print(res)
    print("DESC elapsed: ", t.secs)


def time_nx(filename):
    g = graph_manip.create_networkx_graph(filename)
    with Timer() as t:
        res = nx.algorithms.clique.find_cliques(g)
    print("nx elapsed: %s", t.secs)
    show = False
    if show:
        print("cliques:")
        for x in res:
            print(x)


def post_process_log(logfile):
    """
    For debugging or tracing purposes, clean up a log file for easier examination
    :param logfile:
    :return:
    """
    fp = open(logfile, 'r')
    for l in fp.readlines():
        if (not l.startswith('INFO')) and (not l.startswith('DEBUG')):
            print(l.strip('\n'))


def basic_check(filename):
    g = graph_manip.create_networkx_graph(filename)

    return compare_maximal(g)


def main():
    graph_file = sys.argv[1]
    return basic_check(graph_file)


if __name__ == "__main__":
    #post_process_log(sys.argv[1])
    #time_ours(sys.argv[1])
    print(main())
    #time_nx(sys.argv[1])
    #time_compare(sys.argv[1])