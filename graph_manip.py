"""
Graph manipulation & read in from files.
I investigated (on 5/21/15) the Python graph packages, finding the 3 main ones to be
igraph, networkX and graph-tool. NetworkX appears to be most used but not as high-performance
as graph-tool, but the latter turned out to be challenging to install, so using networkx
"""
import networkx as nx
import itertools


def read_Dimacs(filename):
    """
    Create an adjacency matrix from a DIMACS format graph file.
    :param filename:
    :return: Adjaceny matrix as a dictionary
    """
    g = create_networkx_graph(filename)

    return adj_from_g(g)


def adj_from_g(g):
    """
    Create an adjacency matrix given a networkx formatted graph.
    :param g:
    :return: A as a dictionary
    """
    adj = dict()
    for n in g.nodes():
        adj[int(n)] = set()
        for j in g.neighbors(n):
            adj[int(n)].add(int(j))

    return adj


def create_networkx_graph(filename):
    """
    Create a networkX type graph from a DIMACS format file.
    :param filename:
    :return:
    """
    f = open(filename)
    g = nx.Graph()
    for l in f:
        if l.startswith("e"):
            ch, n1, n2 = l.split()
            g.add_edge(n1, n2)
    return g


def pretty_print(adj):
    n = max(adj.keys())
    out = ' \t'
    for i in range(1, n+1):
        out += str(i) + '\t'
    out += '\n'
    for i in range(1, n+1):
        if len(str(i)) == 1:
            out += str(i) + ' |' + ('\t'*(i+1))
        else:
            out += str(i) + '|' + ('\t'*(i+1))
        if i in adj:
            entry = adj[i]
        else:
            entry = []
        for j in range(i+1, n+2):
            if j in entry:
                out += '1\t'
            else:
                out += '\t'
        out += '\n'
    return '\nAdj Matrix:\n%s', out


def row_wise_order(M, I):
    result = []
    for i, k in itertools.combinations(I, 2):
        # shouldn't need i!=k check, but keeping for now
        if M.has_entry(i, k) and i != k:
            result.append((i, k))
    return result


def get_new_entries(s, d):
    res = set()
    for ele in s:
        res.add(d[ele])
    return res


def count_and_sort(a):
    """
    Sort the keys in the matrix.
    :param a:
    :return: number of keys in the matrix
    """
    sk = list(a.keys())
    sk.sort()
    n = len(sk)
    return n, sk


def map_a_perm(a, p, sk):
    new_adj = dict()
    this_map = {}
    for i in range(len(p)):
        this_map[i + 1] = p[i]
    for i, ele in enumerate(sk):
        newnode = p[i]
        new_adj[newnode] = get_new_entries(a[ele], this_map)
    return new_adj


# permute adj. matrix entries in all possible ways.
def all_permutations(a):
    n, sk = count_and_sort(a)
    result = []
    node_perms = itertools.permutations(sk)
    for p in node_perms:
        result.append(map_a_perm(a, p, sk))
    return result


# change our adj list rep back to a networkx graph
def convert_to_nx(a):
    g = nx.Graph()
    for n in a:
        for m in a[n]:
            g.add_edge(str(n), str(m))
    return g

