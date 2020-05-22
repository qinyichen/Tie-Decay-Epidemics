import numpy as np
import tie_decay_epidemics as tde

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix


def test_tie_decay_graph():
    nodes = np.array(["A", "B", "C"])
    graph = tde.TieDecay_Graph(nodes)
    graph.update_tie_strength("A", "B", 10)
    graph.update_tie_strength("B", "C", 10)

    expected = np.array([[           0, np.exp(-0.1),            0],
                         [np.exp(-0.1),            0, np.exp(-0.1)],
                         [           0, np.exp(-0.1),            0]])

    assert_array_equal(graph.adj, expected)
