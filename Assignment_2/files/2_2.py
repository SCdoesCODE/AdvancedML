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
from sympy.utilities.iterables import multiset_permutations
import itertools





def calc_s(u,i,theta,beta,tree_topology):

    """

    The probability of of the observations below the node u given that the node u has the value i. p(O_b | x_u = i)

    """

    #The two child nodes of u
    if u in tree_topology:
        X_c1,X_c2 = np.where(tree_topology == u)[0]
    #If u is a leaf : return 1 if i is the observed value, otherwise 0
    elif beta[u] == i:
        return 1
    else:
        return 0
    

    prob_sum1 = 0
    prob_sum2 = 0
    for j in range(5):
        #prob of observations below first child of u given that u has the value j, store for efficiency
        if not math.isnan(s_table[X_c1][j]):
            s_c1_j = s_table[X_c1][j]
        else:
            s_c1_j = calc_s(X_c1,j, theta, beta, tree_topology)
            s_table[X_c1][j] = s_c1_j
        #prob of child 1 having value j given u having value i
        p_c1_j_u_i = theta[X_c1][i][j]

        #prob of observations below second child of u given that u has the value j
        if not math.isnan(s_table[X_c2][j]):
            s_c2_j = s_table[X_c2][j]
        else:
            s_c2_j = calc_s(X_c2,j, theta, beta, tree_topology)
            s_table[X_c2][j] = s_c2_j
        #prob of child 2 having value j given u having value i

        p_c2_j_u_i = theta[X_c2][i][j]
        
        prob_sum1 += s_c1_j * p_c1_j_u_i 
        prob_sum2 += s_c2_j * p_c2_j_u_i
        
    
    return prob_sum1*prob_sum2


def brute_force(theta,beta,tree_topology):



    #https://www.cs.cmu.edu/~ninamf/courses/601sp15/slides/11_GrMod1_2-18-2015.pdf

    n_unobserved = 0

    for idx,i in enumerate(beta):
        if math.isnan(i):
            n_unobserved +=1

    k_range = np.arange(5)

    unobserved_perms = list(itertools.product(k_range,repeat = n_unobserved))

    print(unobserved_perms)

    likelihood = 0

    for idx,perm in enumerate(unobserved_perms):
        p_temp = 1
        for i in range(len(beta)):
            #if it has no parents (only root node)
            if math.isnan(tree_topology[i]):
                p_temp *= theta[i][perm[0]]
            #if it has a parents
            else:
                par = int(tree_topology[i])
                #and have not been observed
                if math.isnan(beta[i]):
                    p_temp *= theta[i][perm[par]][perm[i]]
                #and has been observed
                else:
                    p_temp *= theta[i][perm[par]][int(beta[i])]
                    
                
        likelihood += p_temp

    
    
    print("BRUTE FORCE LIKELIHOOD : ",likelihood)





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

    

    # TODO Add your code here

    
    #brute_force(theta,beta,tree_topology)
    


   #starting at the root node and only doing calc_s

    likelihood = 0
    
    for i in range(5):

        likelihood += calc_s(0,i,theta,beta,tree_topology)*theta[0][i]


    return likelihood

s_table = []


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    #filename = "data/q2_2/q2_2_small_tree.pkl"  
    #filename = "data/q2_2/q2_2_medium_tree.pkl"
    filename = "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        
        beta = t.filtered_samples[sample_idx]
        global s_table
        n_nodes = len(beta)
        s_table = np.ones((n_nodes,5))*np.nan
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
