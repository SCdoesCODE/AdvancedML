""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.

    Note: The alphabet "K" is K={0,1,2,3,4}.
"""

import numpy as np
from Tree import Tree
from Tree import Node
import math


def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.


    """

    """

    The value of each item in the tree_topology array is the id 
    of its parent. e.g. [nan  0.  0.  1.  1.] means that we first
    have our root node (has no parent) and it has two children
    1 and 2, and then only child nr.1 has two children. 

    Because we have 5 nodes in this example, we will have
    5 CPD-tables for each of these nodes. The root node 
    will only have one row in its table and the rest will
    be K*K large. e.g. In the CPD-table of v_4 whose parent
    is v_1 we have a K*K grid where cell (i,j) tells us
    the probability of v_1 = i and v_4 = j. 

    e.g. if K=3 and there are 3 leaves then all possible 
    observations are {000,001,002,010,011,012,...,222}
    But we don't want the likelihood of all these possible
    observations. In the main code we have been given 5
    of these possible observations (betas/samples) out of 
    all possible. Our task is to calculate the likelihood
    of each of these separately. We only know the values 
    for the leaves, so we need to consider all possible
    values for the internal nodes in our likelihood estimation.
    So basically if we have beta = [nan nan 3 3 1] we want to
    calc p(X2 = 3, X3 = 3, X4 = 1 | theta, T). Each edge has
    a conditional probability distribution theta_v of the nodes
    value given its parent's value. The value of a random
    variable is often denoted with a lower case letter, e.g. x_1.
    E.g. theta_v = p(X_v|X_pa(v)) = x_pa(v). As we saw in
    mod 6 with the random variables difficulty, intelligence, grade,
    sat, letter, they all had tables describing the conditional probs.
    E.g. grade had difficulty and intelligence as parents,  and the cpd
    table showed all possible combinations of these two. In this
    problem 2.2 all nodes can have values in [K]. 
    """

    """
  
    We know which nodes are leaf-nodes, because they are given by all non-nan 
    entries in beta

    """

    # TODO Add your code here

    leaf_idx = []

    for i in range(len(beta)):
        if not math.isnan(beta[i]):
            leaf_idx.append(i)


    

    O_cond_prob_given_parent = []

    for i in leaf_idx:
        
        O_cond_prob_given_parent.append(theta[i][int(beta[i])][int(tree_topology[i])])

    likelihood = np.product(O_cond_prob_given_parent)
    
    print(likelihood)


    
    

    """
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    print("Calculating the likelihood...")
    likelihood = np.random.rand()
    # End: Example Code Segment

    """

    return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    filename = "data/q2_2/q2_2_small_tree.pkl"  # "data/q2_2/q2_2_medium_tree.pkl", "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
