import graph_manip
import logging
from collections import *
import sys


## Levels:  DEBUG, INFO, WARNING (default), ERROR, CRITICAL
logging.basicConfig(level=logging.WARNING)  # set logging level & to console
#logging.basicConfig(filename='outputD.log', level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename='output.log', level=logging.INFO)
#logging.basicConfig(level=logging.INFO)


class CPattern:
    """
    A Clique pattern consists of a core (q*) and a set of generator sets (Q#).
    Q# is really a set of sets, but sets can't be elements of sets in Python!
    So it's implemented as a list of sets.
    Frozenset doesn't make sense since we're manipulating them; perhaps a bitset?
    For now it's implemented as a list of sets.
    """
    def __init__(self):
        self.q_star = set()
        self.q_sharp = []

    def __repr__(self):
        return "(" + str(list(self.q_star)) + " " + str(self.q_sharp) + ")"

    def __str__(self):
        return "(" + str(list(self.q_star)) + " " + str(self.q_sharp) + ")"

    def __lt__(self, other):
        return self.clique_size() < other.clique_size()

    def __eq__(self, other):
        if self.q_star == other.q_star:
            if self.get_gen_nodes() == other.get_gen_nodes():
                if len(self.q_sharp) == len(other.q_sharp):
                    for gen in self.q_sharp:
                        if gen not in other.q_sharp:
                            return False
                    for gen in other.q_sharp:
                        if gen not in self.q_sharp:
                            return False
                    return True
        return False

    def is_null(self):
        return (not self.q_star) and (not self.q_sharp)

    def clique_size(self):
        return len(self.q_star) + len(self.q_sharp)

    def add_to_core(self, nodes):
        self.q_star = self.q_star.union(nodes)

    def rem_from_core(self, nodes):
        for node in nodes:
            self.q_star.discard(node)

    def core(self):
        return self.q_star

    def gen(self):
        return self.q_sharp

    def add_to_gen(self, nodes):
        self.q_sharp.extend([set(nodes)])

    def update_gen(self, position, nodes):
        #  update an existing generator set element with nodes
        curr = self.q_sharp.pop(position)
        new_gen = curr.union(nodes)
        self.q_sharp.extend([new_gen])

    def rem_from_gen(self, nodes):
        for n in nodes:
            pos = self.gen_position(n)
            if pos != -1:
                self.q_sharp[pos].remove(n)

    def rem_from_ith_gen(self, nodes, pos):
        for n in nodes:
            if n in self.q_sharp[pos]:
                self.q_sharp[pos].remove(n)

    def getUnion(self):
        # return all the nodes in the pattern
        result = self.q_star.union(self.get_gen_nodes())
        return result

    def get_gen_nodes(self):
        result = set()
        for g in self.q_sharp:  # a list of sets
            result = result.union(g)
        return result

    def nth_gen_nodes(self, n):
        return self.q_sharp[n]

    # find the position of n in generator set, if any
    def gen_position(self, n):
        for i, ele in enumerate(self.q_sharp):
            if n in ele:
                return i
        return -1


# TODO: The Projector matrix entries themselves could be implemented as
#  dictionaries as well for efficiency (time & space).
#  Also, this whole algorithm may break if node names are not ints.
class ProjMatrix:
    """
    Projector Matrix, created from an adjacency matrix.
    NOTE: hack: we have a zero entry even though the examples we've worked all start indexing from 1
    """
    def __init__(self, adj):
        #plus 1 since graphs index from 1 not 0
        numNodes = max(list(adj.keys())) + 1
        p = dict()
        sk = list(adj.keys())
        sk.sort()
        for j in range(numNodes):
            p[j] = [set()]*numNodes
        for j in sk:
            adjn = adj[j]
            for k in sk:
                if k > j:
                    in_both = adjn.intersection(adj[k])
                    p[j][int(k)] = in_both

        self.matrix = p
        self.linked = defaultdict(list)  # calculate in find_trivial
        self.trivial = []  # calculate on demand only

    def find_trivial(self, a):
        """Find all trivial cliques that can't be extended,
        record them, and save the modified projector matrix.
        """
        sk = list(a.keys())
        sk.sort()
        nodes = set(self.matrix.keys())
        for i in sk:
            adj = a[i]
            for d in nodes.difference(adj):
                self.matrix[i][d] = set()
        # store all the nodes linked to another one (upper & lower)
        linked = defaultdict(list)
        p = self.matrix
        for e in p:
            for i, b in enumerate(p[e]):
                if b:
                    linked[e] += [i]
                    linked[i] += [e]
        self.linked = linked

        trivial = []
        for n in nodes:
            if n in a:
                adj = a[n]
                for j in adj:
                    #consider only upper of adj
                    if n < j and not self.matrix[n][j]:
                        trivial.append((n, j))
        self.trivial = trivial
        logging.debug("\n   trivial cliques: %s", str(trivial))

    def clique_nodes(self):
        """
        :return: I, all nodes that might be part of some non-trivial clique.
        """
        result = set()
        for j in self.matrix:
            for k in self.matrix[j]:
                result = result.union(k)

        return result

    #TODO: could combine this with get_entry and return None if entry is empty
    def has_entry(self, i, k):
        lower = min(i, k)
        higher = max(i, k)
        return self.matrix[lower][higher]

    def get_entry(self, i, k):
        lower = min(i, k)
        higher = max(i, k)
        return self.matrix[lower][higher]

    #Print from index 1 due to 0 hack!
    def __str__(self):
        out = ' \t '
        n = max(self.matrix.keys())
        for i in range(1, n+1):
            out += str(i) + '\t\t'
        out += '\n'
        for i in range(1, n):
            if len(str(i)) == 1:
                out += str(i) + ' |' + ('\t'*(2*i+1))
            else:
                out += str(i) + '|' + ('\t'*(2*i+1))
            if i in self.matrix:
                entry = self.matrix[i]
            else:
                entry = [set()] * (n + 1)
            for j in range(i+1, n+1):
                if entry[j]:
                    if len(str(entry[j])) <= 6:
                        tab = '\t\t'
                    else: tab = '\t'
                    out += str(entry[j]) + tab
                else:
                    out += '{}\t\t'
            out += '\n'
        return out


class Templates:
    """ Information needed for some of the DESC actions.
    """
    def __init__(self):
        self.row = -1
        self.newGen = None  # the one of i,k that we might have to move to a Q#
        self.q_sharp_pos = None  # the position of the Q# matching node (i,k), if any

    def get_row(self, i, k, pattern, Pu):
        """
        Lookup the status of the Base wrto the pattern passed in.
    :param i: Proj matrix "row"
    :param k: Proj matrix "col"
    :param pattern:
    :param Pu: union of all pattern's nodes
    :return: int indicating row number from DESC table
    """
        if (i not in Pu) and (k not in Pu):
            self.row = 1
            logging.info('\ni = %d not in Pu and k = %d not in Pu', i, k)
            return 1
        q_star = pattern.core()
        q_sharp_nodes = pattern.get_gen_nodes()
        if i in q_star and k in q_star:
            self.row = 2
            logging.info('\ni = %d in q* and k = %d in q*', i, k)
            return 2
        if i in q_sharp_nodes and k in q_sharp_nodes:
            self.row = 3
            logging.info('\ni = %d in q# and k = %d in q#', i, k)
            return 3
        if ((i not in Pu) and k in q_star) or (i in q_star and (k not in Pu)):
            self.row = 4
            if i not in Pu:
                self.newGen = i
                logging.info('\ni = %d not in Pu and k = %d in q*', i, k)
            else:
                self.newGen = k
                logging.info('\ni = %d in q* and k = %d in not in Pu', i, k)
            return 4
        if ((i not in Pu) and k in q_sharp_nodes) or (i in q_sharp_nodes and (k not in Pu)):
            if k in q_sharp_nodes:
                self.q_sharp_pos = pattern.gen_position(k)
                self.newGen = i
                logging.info('\ni = %d not in Pu and k = %d in q#', i, k)
            else:
                self.q_sharp_pos = pattern.gen_position(i)
                self.newGen = k
                logging.info('\ni = %d in q# and k = %d not in Pu', i, k)
            self.row = 5
            return 5
        elif (i in q_star and k in q_sharp_nodes) or (i in q_sharp_nodes and k in q_star):
            i_gen_pos = pattern.gen_position(i)
            if i_gen_pos == -1:
                k_gen_pos = pattern.gen_position(k)
                self.newGen = i
                self.q_sharp_pos = k_gen_pos
                logging.info('\ni = %d in q* and k = %d in q#', i, k)
            else:
                self.newGen = k
                self.q_sharp_pos = i_gen_pos
                logging.info('\ni = %d in q# and k = %d in q*', i, k)
            self.row = 6
            return 6
        logging.warning("ERR: no row matched in getRow: (%d, %d), pattern: %s, Pu: %s", i, k, pattern, Pu)


def action3(p, c_ik, i, k):
    new_nodes = c_ik.difference(p.getUnion())
    logging.debug("\n   Pnew: %s", new_nodes)
    if len(new_nodes) == 1:  # new triangle
        newp = CPattern()
        newp.add_to_core(new_nodes.union({i, k}))
        logging.debug("\n   New pattern: %s", str(newp))
        return newp, True
    else:
        logging.debug("\n   Orig p: %s", str(p))
        core_int = c_ik.intersection(p.core())  # M
        if len(core_int) == 1:
            p.rem_from_core(core_int)
            p.add_to_gen(core_int.union(new_nodes))
            logging.debug("\n   Updated p: %s", str(p))
        else:
            p.add_to_gen(new_nodes)
            logging.debug("\n   Updated p: %s", str(p))
        return p, False


def process_projs(c_ik, i, k, F):
    """
    Process the Projectors.
    :param c_ik:
    :param i:
    :param k:
    :param F: current patterns
    :return: updated patterns
    """
    newpatterns = []
    for p in F:
        if c_ik.difference(p.getUnion()):
            if i in p.core() and k in p.core():
                logging.info("\nAction 3")
                patt, new = action3(p, c_ik, i, k)
                if new:
                    newpatterns.append(patt)
            elif i in p.gen() and k in p.gen():
                logging.info("\nAction 3")
                patt, new = action3(p, c_ik, i, k)
                if new:
                    newpatterns.append(patt)
            elif (i in p.core() and k in p.gen()) \
                    or (i in p.gen() and k in p.core()):
                logging.info("\nAction 3")
                patt, new = action3(p, c_ik, i, k)
                if new:
                    newpatterns.append(patt)
    F.extend(newpatterns)
    return F


def new_nodes(c_ik, Fu):
    rem = c_ik.difference(Fu)
    logging.debug("\n    Cik \ Fu: %s", str(rem))
    return rem


def clique_check(pattern, T):
    """
    Does pattern accurately reflect T as if it were a clique?
    :param pattern:
    :param T: set of nodes representing triangles
    :return: Boolean
    """
    Pu = pattern.getUnion()
    if not T.issubset(Pu):
        return False
    if not pattern.gen():
        return T == Pu
    #if there are nodes in the core not in T, fail
    if pattern.core().difference(T):
        return False
    expect_in_gens = T.difference(pattern.core())
    for g in pattern.q_sharp:
        in_this = expect_in_gens.intersection(g)
        if len(in_this) != 1:
            return False
        expect_in_gens= expect_in_gens.difference(in_this)
    if expect_in_gens:
        return False
    return True


def ik_in_core(i, k, pattern):
    core = pattern.core()
    return i in core and k in core


def adjust_patterns(F, T, i, k):
    """
    Adjust the patterns.
    :param F: all current patterns
    :param T: triangles of current projector matrix entry
    :return: (possibly) modified F and a flag indicated whether F was modified
    """
    Fu = all_nodes(F)
    Missing = Fu.difference(T)
    logging.debug("\n    Missing (Fu\T): %s", str(Missing))
    #check if T represented by a pattern
    for p in F:
        if not p.getUnion().difference(T):
            logging.debug("MissingP empty: %s", str(p))
            return F
        if clique_check(p, T):
            logging.debug("\n   Clique check passed: %s", str(p))
            return F
        if triangles_covered(T.difference({i, k}), i, k, p):
            logging.debug("\n   Covered triangles passed: %s", str(p))
            return F
    for patt in F:
        Miss_projectors = patt.getUnion().difference(T)
        Miss_Star = Miss_projectors.intersection(patt.core())
        t = Templates()
        row = t.get_row(i, k, patt, patt.getUnion())
        if row == 2:
            logging.info("\nAction 4")
            logging.debug("\n   Orig pattern: %s", str(patt))
            logging.debug("\n   Missing*: %s", str(Miss_Star))
            patt.rem_from_core(Miss_Star)
            if len(Miss_Star) > 1:
                patt.add_to_gen(Miss_Star)

            logging.debug("\n   Updated pattern: %s", str(patt))
        elif row == 4:
            logging.info("\nAction 5")
            if Miss_projectors:
                logging.debug("\n   Orig patt: %s", str(patt))
                patt.rem_from_core(Miss_projectors)
                # see if they are all in the same q#
                posits = []
                for node in Miss_projectors:
                    posits.append(patt.gen_position(node))
                gens = set(posits).difference({-1})
                if len(gens) == 1:
                    patt.update_gen(gens.pop(), Miss_projectors.union({t.newGen}))
                else:
                    patt.add_to_gen(Miss_projectors.union({t.newGen}))
                logging.debug("\n    Updated pattern: %s", str(patt))
        elif row == 5:
            logging.info("\nAction 6")
            nodes_in_other = patt.nth_gen_nodes(t.q_sharp_pos)
            if Miss_projectors.difference(nodes_in_other):
                logging.debug("\n   Orig patt: %s", str(patt))

                move_candidate = Miss_projectors.difference(nodes_in_other)
                patt.rem_from_core(move_candidate)
                # see if they are all in the same q#
                posits = []
                for node in move_candidate:
                    posits.append(patt.gen_position(node))
                gens = set(posits).difference({-1})
                if len(gens) == 1:
                    patt.update_gen(gens.pop(), move_candidate.union({t.newGen}))
                else:
                    patt.add_to_gen(move_candidate.union({t.newGen}))
                logging.debug("\n   Updated pattern: %s", str(patt))
        elif row == 6:
            logging.info("\nAction 7")
            logging.debug("\n   Orig patt: %s", str(patt))
            patt.rem_from_core(Miss_Star)
            if Miss_Star:
                patt.update_gen(t.q_sharp_pos, Miss_Star)
                logging.debug("\n   Updated patt: %s", str(patt))
    return F


def subset_core(T, pattern):
    return not pattern.gen() and T.issubset(pattern.core())


def triangles_covered(c_ik, i, k, pattern):
    """
    Are all triangles represented in pattern?
    :param c_ik:
    :param i:
    :param k:
    :param pattern:
    :return: Boolean
    """
    for n in c_ik:
        this_triangle = {i, k, n}
        if not clique_check(pattern, this_triangle):
            return False
    return True


def process_base(c_ik, F, i, k):
    """
    Process the Base.
    :param c_ik:
    :param F:
    :param i:
    :param k:
    :return: Tuple: updated F, Boolean indicating whether F changed
    """
    T = c_ik.union({i, k})
    for p in F:
        if clique_check(p, T):
            return F, False
        if triangles_covered(c_ik, i, k, p):
            return F, False
    for pattern in F:
        Pu = pattern.getUnion()
        t = Templates()
        row = t.get_row(i, k, pattern, Pu)
        Jold_star = pattern.core().intersection(c_ik)
        if row == 1:
            logging.info("\nAction 1")
            newp = CPattern()
            newp.add_to_core(c_ik.union({i, k}))
            logging.debug("\n   New Pattern: %s", str(newp))
            F.append(newp)
            return F, True
        elif row in [4, 5]:
            if len(c_ik) == 1 or \
                    (Jold_star and len(pattern.core()) <= 3):
                logging.info("\nAction 2")
                logging.debug("\n    c_ik: %s", str(c_ik))
                newp = CPattern()
                newp.add_to_core(c_ik.union({i, k}))
                logging.debug("\n   New Pattern: %s", str(newp))
                F.append(newp)
                return F, True
        elif row == 6:
            if Jold_star and len(pattern.core()) < 3:
                logging.info("\nAction 2'")
                logging.debug("\n    Jold*: %s", str(Jold_star))
                newp = CPattern()
                newp.add_to_core(c_ik.union({i, k}))
                logging.debug("\n   New Pattern: %s", str(newp))
                F.append(newp)
                return F, True

    return F, False


def new_pattern_needed(F, i, k):
    """ True if we need to Process the Base (i,k).
    Boolean function: True if i and k NOT both in the core of the same pattern.
    within F
    :param F: Patterns
    :param i:
    :param k:
    :return: Boolean
    """
    for p in F:
        if i in p.core() and k in p.core():
            logging.debug("\n   NO Need to process T13-17")
            return False
    logging.debug("\n   DO Need to process T13-17")
    return True


def all_nodes(F):
    """
    :param F: Current set of patterns.
    :return: Union of all the nodes in the patterns.
    """
    results = set()
    for p in F:
        results = results.union(p.getUnion())
    return results


def update_patterns(F, c_ik, i, k):
    """
    Heart of the DESC algorithm.
    :param F:
    :param c_ik:
    :param i:
    :param k:
    :return: Updated set of patterns.
    """
    T = c_ik.union({i, k})
    logging.info("\ni=%d, k=%d, C_ik=%s", i, k, str(c_ik))
    Fu = all_nodes(F)
    logging.debug("\n  Fu=%s", str(Fu))
    newF = False
    if new_pattern_needed(F, i, k):
        F, newF = process_base(c_ik, F, i, k)
    Fu = all_nodes(F)
    if new_nodes(c_ik, Fu):
        F = process_projs(c_ik, i, k, F)
    # Adjust patterns
    if not newF or all_nodes(F).difference(T):
        F = adjust_patterns(F, T, i, k)

    return F


def ordered_search(srtd_projs, M):
    """
    Iterate through M in the (i,k) order specified by srtd_projs
    :param srtd_projs: list of (i,k) pairs that index into M
    :param M: Projector matrix
    :return:
    """
    F = []
    for i, k in srtd_projs:
        logging.info("\n\nF = %s", str(F))
        c_ik = M.get_entry(i, k)
        if not F:
            # just for logging purposes
            logging.debug("\ni = %d   k = %d\tC_ik = %s", i, k, str(c_ik))
            new_p = CPattern()
            new_p.add_to_core(c_ik.union({i, k}))
            F = [new_p]
            logging.info("\nFirst pattern")
            logging.debug("\n   F = %s", str(F))
        else:
            F = update_patterns(F, c_ik, i, k)

    logging.info("\nDONE, F: %s", str(F))
    return F


def iter_order(pm, I):
    return graph_manip.row_wise_order(pm, I)


def find_clique_patterns(M, I):
    """
    Search for clique patterns by finding relevant entries in M (via iter_order),
    and then searching through them (via ordered_search).
    :param M: projector matrix
    :param I: Set of nodes that may contain a non-trivial maximal clique
    :return: Clique patterns
    """
    srtd_projs = iter_order(M, I)

    return ordered_search(srtd_projs, M)


def get_pm_I(adj):
    """
    Perform preprocessing steps.
    :param adj:
    :return:
    """
    proj_matrix = ProjMatrix(adj)
    proj_matrix.find_trivial(adj)
    logging.info("\nprojMatrix:\n%s", str(proj_matrix))

    I = proj_matrix.clique_nodes()
    return proj_matrix, I


def processing_steps(adj):
    proj_matrix, I = get_pm_I(adj)
    return proj_matrix.trivial, find_clique_patterns(proj_matrix, I)


def process_graph_file(f_ptr):
    """
    Create an adjacency matrix from a graph file
    :param f_ptr: file name
    :return: Adjacency matrix
    """
    return graph_manip.read_Dimacs(f_ptr)


def process_from_g(g):
    adj = graph_manip.adj_from_g(g)
    return processing_steps(adj)


def process_from_file(graph_file):
    adj_matrix = process_graph_file(graph_file)
    logging.info(graph_manip.pretty_print(adj_matrix))
    return processing_steps(adj_matrix)


def main():
    graph_file = sys.argv[1]
    return process_from_file(graph_file)


if __name__ == "__main__":
    res = main()
    print(res)