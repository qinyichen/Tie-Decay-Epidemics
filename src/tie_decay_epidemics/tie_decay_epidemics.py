from __future__ import division
import numpy as np
import random
import time
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from numpy import linalg as LA
from utils import dataframe_to_dict

# TODO: Changed the input format of the edgelist to a dataframe --> Test if it works.

class TieDecay_Graph(object):
    """A tie-decay network with decay coefficient alpha at a particular time.

    Parameters
    ----------
    nodes : 1darray
         Nodes of the tie-decay network.
    time : float
         The current time stamp of the tie-decay network.
    alpha : float
         Decay coefficient of the tie-decay network.
    init_adj : 2darray, optional
         The initial strength (adjacency) matrix of the tie-decay network at
         time t = 0. This matrix should be undirected.
    """
    def __init__(self, nodes, time=0, alpha=0.01, init_adj=None):
        self.alpha = alpha
        self.nodes = nodes
        self.time = time

        # Use the initial adjacency matrix if provided
        # Otherwise we start with a zero adjacency matrix
        if init_adj is None:
            self.adj = csr_matrix((len(nodes), len(nodes)), dtype=np.float_).toarray()
        else:
            self.adj = init_adj.toarray()

    def get_node_id(self, node):
        """Returns the index of the particular node."""
        return np.nonzero(self.nodes == node)[0][0]

    def update_tie_strength(self, src, dst, time, weight=1):
        """Update the tie strength given an interaction.

        Note:
        (1) Update self.time after all the updates of tie strengths are done
        (2) Graph is undirected---each interaction affects two terms in adj

        Parameters
        ----------
        src : int
             the source node of the interaction
        dst : int
             the destination node of the interaction
        time : float
             the time at which the interaction takes place
        weight : float
             the amount by which the tie strength will be incremented
        """
        weight_add = weight*np.exp(self.alpha*(self.time - time))
        self.adj[self.get_node_id(src), self.get_node_id(dst)] += weight_add
        self.adj[self.get_node_id(dst), self.get_node_id(src)] += weight_add


class TieDecay_SIS(object):
    """
    Simulating an SIS process on a tie-decay network.

    Parameters
    ----------
    nodes : 1darray
         Nodes of the tie-decay network.
    infected : 1darray
         Nodes that are initially infected in the tie-decay network.
    edgelist : dataframe
         The series of interactions between nodes/agents. The edgelist will
         be converted to a dict in __init__().
    rateSI : float
         The (maximum) rate of infection.
    rateIS : float
         The rate of recovery.
    alpha : float
         Decay coefficient of the tie-decay network.
    init_adj : 2darray, optional
         The initial strength (adjacency) matrix of the tie-decay network at
         time t = 0. This matrix should be undirected.
    have_system_matrix : bool, optinal
         A flag indicating whether the system matrix should be computed.
    system_matrix_period : int, optional
         The length of the period in which the system should be computed.
    verbose : bool, optional
         A flag indicating whether the infection/recovery events will be printed.
    """

    def __init__(self,
                 nodes, infected, edgelist, rateSI, rateIS, alpha,
                 init_adj=None,
                 have_system_matrix=True,
                 system_matrix_period=0,
                 verbose=True):

        self.graph = TieDecay_Graph(nodes, alpha=alpha, init_adj=init_adj)
        self.nodes = nodes
        self.time = 0
        self.verbose = verbose

        # Each edge has an infection probability = rateSI * max(B_{ij}, 1)
        self.rateSI = rateSI
        self.rateIS = rateIS

        # Node idxs that are infected in the beginning
        self.init_infected = set([self.graph.get_node_id(n) for n in infected])

        # Node idxs that are currently infected/susceptible
        self.infected = self.init_infected.copy()
        self.susceptible = set(range(len(nodes))).difference(self.infected)

        self.infected_history = [self.get_infected()]
        self.susceptible_history = [self.get_susceptible()]
        self.reproduction_number = 0

        # We convert the edgelist dataframe to a dict to better access the data
        self.edgelist = dataframe_to_dict(edgelist)

        # System matrix computations
        self.have_system_matrix = have_system_matrix
        self.system_matrix_period = system_matrix_period

        if self.have_system_matrix:
            self.system_matrix = \
                    (1-rateIS)*np.identity(len(self.nodes))+\
                    rateSI*np.where(self.graph.adj>1, 1, self.graph.adj)
            # w = LA.eigvals(self.system_matrix)
            # self.critical_values = [max(abs(w))]
            w = eig(self.system_matrix)
            self.critical_values = [max(abs(w))]
            print("t = {}, critical value is {}".format(self.time, max(abs(w))))

    def update_graph(self, t):
        """Update the state of the tie-decay network to the current time t.

        Parameters
        ----------
        t : float
            Read in every interaction in the edgelist till time t. Update the
            attributes of the tie-decay network accordingly.
        """
        time_updated_until = self.graph.time

        # Reset the graph time to be the current time
        self.graph.time = t

        # Note: CHANGE THE WAY TIE STRENGTH IS DEFINED HERE --- CAP = 1
        # Also update all the existing weights to reflect decay of ties
        self.graph.adj = self.graph.adj * \
                            np.exp(self.graph.alpha * (time_updated_until-t))

        # Add in any new iterations
        for time in range(time_updated_until+1, t+1):
            try:
                edges = self.edgelist[time]
                for edge in edges:
                    self.graph.update_tie_strength(edge[0], edge[1], edge[2])
            except Exception as e:
                # if no interactions take place at t
                pass

        __adj__ = np.where(self.graph.adj>1, 1, self.graph.adj)

        # Update the have_system_matrix matrix if needed
        if self.have_system_matrix and t <= self.system_matrix_period:
            self.system_matrix = np.matmul(\
                        ((1-self.rateIS)*np.identity(len(self.nodes))+\
                        self.rateSI*__adj__), self.system_matrix)
            w, vl, vr = eig(self.system_matrix, left=True, right=True)

            # Note: the power should be (# of temporal snapshots)^-1
            critical_value = max(abs(w))**(1/t)
            self.critical_values.append(critical_value)
            print("t = {}, critical value is {}".format(t, critical_value))


    def run(self, max_time):
        """Perform an SIS simulation on the tie-decay network.

        Note: if max_time > # of temporal snapshots, the spreading will continue
        to take place as the adjacency matrix gradually decays to zero.

        Parameters
        ----------
        max_time : float
            The maximum time of the SIS process.
        """
        while len(self.infected) > 0 and self.time <= max_time:
            self.time += 1
            self.update_graph(self.time)

            for node_idx in self.susceptible.copy():
                self.infection_event(node_idx)

            for node_idx in self.infected.copy():
                self.recovery_event(node_idx)

            self.susceptible_history.append(self.get_susceptible())
            self.infected_history.append(self.get_infected())

            if self.verbose:
                print("Step {}: Proportion of two types: {:.3f} {:.3f}" \
                    .format(self.time, self.get_susceptible()/len(self.nodes),
                        self.get_infected()/len(self.nodes)))

        if self.verbose:
            print("Outbreak size: {} out of {}"\
                  .format(len(self.nodes)-self.get_susceptible()), len(self.nodes))
            print("Time of Epidemic Transition: {}".format(self.get_peak_time()))


    def infection_event(self, node_idx):
        """Infect a node by its rate of infection.

        Parameters
        ----------
        node_idx : int
            The index of the node where the infection takes place.
        """
        # Obtain its infected neighbor idxs
        nbrs = self.graph.adj[:, node_idx].nonzero()[0]
        infected_nbrs = self.infected.intersection(set(nbrs))

        # If there are no infected neighbors, no infection event will take place
        if len(infected_nbrs) == 0:
            return

        # Each infection takes place independently with respective probability
        for nbr_idx in infected_nbrs:
            if random.random() < np.min([self.graph.adj[nbr_idx,node_idx],1])*self.rateSI:
                self.infected.add(node_idx)
                self.susceptible.remove(node_idx)

                # If the node is directly infected by the source of infection,
                # increment the basic reproduction number by 1
                if nbr_idx in self.init_infected:
                    self.reproduction_number += 1
                if self.verbose:
                    print("Node {} is infected by {}."\
                        .format(self.nodes[node_idx], self.nodes[nbr_idx]))
                break

    def recovery_event(self, node_idx):
        """A node returns to being susceptible by its rate of recovery.

        Parameters
        ----------
        node_idx : int
            The index of the node where the recovery takes place.
        """
        if random.random() < self.rateIS:
            self.infected.remove(node_idx)
            self.susceptible.add(node_idx)
            if self.verbose:
                print("Node {} is back to susceptible.".format(self.nodes[node_idx]))

    def get_susceptible(self):
        """Return the number of susceptible nodes at a time."""
        return len(self.susceptible)

    def get_infected(self):
        """Return the number of infected nodes at a time."""
        return len(self.infected)

    def get_peak_number(self):
        """Return the peak number of infection."""
        return max(self.infected_history)

    def get_peak_time(self):
        """Return the times when the number of infections reached its peak."""
        return np.where(np.array(self.infected_history) == \
                                            self.get_peak_number())[0][0]

    def get_outbreak_size(self):
        """Return the number of agents infected in history."""
        return len(self.nodes) - self.get_susceptible()
