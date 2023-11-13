"""
Created on Tue Nov 15 13:46:38 2022

@author: rahul.swamy
"""

# Stage 4 in Optimap: A Practical Optimization Framework for Congressional Redistricting in Arizona


import csv
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'serif', 'weight' : 'bold', 'size'   : 30}
plt.rc('font', **font)
import networkx as nx
import random
import os
import pandas as pd
import draw_maps as main
import time
import pickle



def do_popbal_localsearch(map_filename):
    with open(map_filename+'.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rowlist = [row for row in spamreader]
        z_i = {int(rowlist[i][0]): int(rowlist[i][1])  for i in range(1,len(rowlist)) if rowlist[i] != []}
        z_k = {k: [i for i in z_i if z_i[i] == k] for k in range(1,K+1)}


    """ Create hybrid-block map and hybrid-block-group map from hybrid-tract map """
    z_blockgroup_i = {i: z_i[int(i/10)] if i > 9999 else z_i[i]  for i in hybrid_blockgroup_graph.nodes()}
    z_blockgroup_k = {k: [i for i in hybrid_blockgroup_graph.nodes() if z_blockgroup_i[i] == k] for k in range(1,K+1)}

    """ Do local search on hybrid block map to improve population balance """
    start_time_blockimprovement = time.time()
    print("popbal before is:", main.metric.evaluate_population_balance(z_blockgroup_k,hybrid_blockgroup_graph))

    pop_bal = main.metric.evaluate_population_balance(z_k, hybrid_tract_graph)

    ub_compactness_needed = main.metric.evaluate_compactness_edgecuts(z_i, hybrid_tract_graph)
    # ub_hier_cmpttv_needed = - len(main.metric.evaluate_number_of_cmptitv_dists(z_k, hybrid_tract_graph, 0.07)[0]) + 1  #max_margin
    num_cmpttv, max_margin_non_cmmptv = main.metric.evaluate_hierarchical_competitiveness(z_k,hybrid_tract_graph,0.07)
    ub_hier_cmpttv_needed = - num_cmpttv + max_margin_non_cmmptv

    print("Upper bound on ub_hier_cmpttv_needed: %f, ub_compactness_needed: %f"%(ub_hier_cmpttv_needed,ub_compactness_needed))

    """ Do local search on hybrid block-group map to improve population balance """
    z_k_after_bg, z_i_after_bg, obj_outer, movement_log = main.algorithm.local_search(z_blockgroup_k, z_blockgroup_i, hybrid_blockgroup_graph, 'pop_bal', {}, print_log=True, ub_hier_cmpttv_needed=ub_hier_cmpttv_needed, ub_compactness_needed=ub_compactness_needed, n_iterations_no_improvement=n_iterations_no_improvement, vickrey_initial=False)

    num_cmpttv, max_margin_non_cmmptv = main.metric.evaluate_hierarchical_competitiveness(z_k_after_bg, hybrid_blockgroup_graph, 0.07)
    compactness = main.metric.evaluate_compactness_edgecuts(z_i_after_bg, hybrid_blockgroup_graph)
    print("Cmpttveness and compactness in block group level",num_cmpttv, max_margin_non_cmmptv,compactness)
    print("popbal after block group improvement is:", main.metric.evaluate_population_balance(z_k_after_bg, hybrid_blockgroup_graph))
    df_movement_bg = pd.DataFrame(movement_log)
    df_movement_bg.to_csv(map_filename+'_popbal=%i_movementlog_blockgroup.csv'%(pop_bal*P_bar), index=False)


    """ Convert hybrid-block-group map to hybrd-block map """

    z_block_i_before = {i:  z_i_after_bg[int(i/1000)] if i > 9999 else  z_i_after_bg[i] for i in hybrid_block_graph.nodes()}
    z_block_k_before = {k: [i for i in hybrid_block_graph.nodes() if z_block_i_before[i] == k] for k in range(1,K+1)}

    num_cmpttv, max_margin_non_cmmptv = main.metric.evaluate_hierarchical_competitiveness(z_block_k_before, hybrid_block_graph, 0.07)
    compactness = main.metric.evaluate_compactness_edgecuts(z_block_i_before, hybrid_block_graph)
    print("Cmpttveness and compactness in block level:",num_cmpttv, max_margin_non_cmmptv,compactness)

    z_k_after, z_i_after, obj_outer, movement_log = main.algorithm.local_search(z_block_k_before, z_block_i_before, hybrid_block_graph, 'pop_bal', {}, print_log=True, ub_hier_cmpttv_needed=ub_hier_cmpttv_needed, ub_compactness_needed=ub_compactness_needed, n_iterations_no_improvement = n_iterations_no_improvement, vickrey_initial=False)

    print("popbal after block level improvement is:", main.metric.evaluate_population_balance(z_k_after,hybrid_block_graph))
    time_blockimprovement = time.time() - start_time_blockimprovement


    """ Plot: Convert HCB map to block graph, plot and get analytics """

    opt_effgap, passymetry_opt, max_margin, majmin, compactness, n_cmpttv, df_solution = main.outlet.print_metrics(z_k_after, z_i_after, hybrid_block_graph,P_bar)
    print(df_solution)

    z_i_block = {i: z_i_after[label_block_community[i]] if i not in z_i_after else z_i_after[i] for i in block_graph.nodes()}
    z_k_block = {k: [i for i in block_graph.nodes() if z_i_block[i] == k] for k in range(1,K+1)}

    pop_bal = main.metric.evaluate_population_balance(z_k_block, block_graph)
    print("popbal of final block map is:", pop_bal*P_bar)

    """ The below commented code visualizes and saves the map as a .csv file """
    #
    # time_pop_improvment = time.time() - start_time_blockimprovement
    #
    # df_solution.to_csv(map_filename+'_popbal=%i_analytics.csv'%(pop_bal*P_bar), index=False)
    # df_movement_block = pd.DataFrame(movement_log)
    # df_movement_block.to_csv(map_filename+'_popbal=%i_movementlog_block.csv'%(pop_bal*P_bar), index=False)
    #
    # metrics_dict = {'majmin':majmin,'EG':opt_effgap,'PA':passymetry_opt,'maxmargin':max_margin, \
    #                 'n_cmpttv':n_cmpttv,'hier_cmpttv':n_cmpttv-max_margin,'compactness':compactness, \
    #                 'pop_bal':pop_bal*P_bar, 'time_pop_improvment':time_pop_improvment}
    # pd.DataFrame.from_dict(data=metrics_dict,orient='index').to_csv(map_filename+'_popbal=%i_metrics.csv'%(pop_bal*P_bar), index=True)
    #
    # with open(map_filename+'_popbal=%i.csv'%(pop_bal*P_bar), 'w') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(['GEOID', 'district'])
    #     for key, value in list(z_i_block.items()):
    #         writer.writerow([key, value])


def read_blockmap_save_analytics(blockmap_filename):
    block_graph = nx.read_gpickle('datasets/block_graph.gpickle')

    with open(blockmap_filename+'.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rowlist = [row for row in spamreader]
        z_i_block = {int(rowlist[i][0]): int(rowlist[i][1])  for i in range(1,len(rowlist)) if rowlist[i] != []}
        z_k_block = {k: [i for i in z_i_block if z_i_block[i] == k] for k in range(1,K+1)}

    z_i_hybridblock = {label_block_community[i] if i in label_block_community else i: z_i_block[i] for i in z_i_block}
    z_k_hybridblock = {k: [i for i in z_i_hybridblock if z_i_hybridblock[i] == k] for k in range(1,K+1)}

    opt_effgap, passymetry_opt, max_margin, majmin, compactness, n_cmpttv, df_solution = main.outlet.print_metrics(z_k_block, z_i_block, block_graph,P_bar)
    print(df_solution)
    print("popbal of final block map is:", pop_bal*P_bar)

    """ Uncomment the following code to save the metrics as csv files """
    # df_solution.to_csv(blockmap_filename+'_analytics.csv', index=False)
    # metrics_dict = {'majmin':majmin,'EG':opt_effgap,'PA':passymetry_opt,'maxmargin':max_margin, \
    #                 'n_cmpttv':n_cmpttv,'hier_cmpttv':n_cmpttv-max_margin,'compactness':compactness, \
    #                 'pop_bal':pop_bal*P_bar}
    # pd.DataFrame.from_dict(data=metrics_dict,orient='index').to_csv(blockmap_filename+'_metrics.csv', index=True)



""" Inputs """
K = 9     # number of districts
n_counties_to_split = 7    # number of counties to split
tau = .01     # population deviation threshold for stages 1 through 3
cmpttv_threshold = .07    # vote-margin within which a district is considered competitive
number_of_majority_minority_dists_needed = 2    # number of majority-minority districts
majmin_threshold = 0.5   # minimum fractional minority population for a district to be considered majority-minority



""" Read data """
county_graph = main.input.read_data_county_2020()
tract_graph = main.input.read_data_tract_2020()
communities_dict, places_dict = main.input.read_communities_of_interest_and_places()
hybrid_tract_graph, label_community_tract, label_tract_community = main.multilevel_algo.create_hybrid_graph(county_graph, tract_graph, n_counties_to_split, communities_dict, places_dict,'tract')
P_bar = int(sum(tract_graph.nodes[i]['pop'] for i in tract_graph.nodes())/K) # average district population
main.P_bar = P_bar

block_graph = nx.read_gpickle('datasets/block_graph.gpickle')
blockgroup_graph = nx.read_gpickle('datasets/blockgroup_graph.gpickle')
print("Number of units in tract, block-group and block graphs:",len(tract_graph),len(blockgroup_graph),len(block_graph))


""" Create hybrid block-group and hybrid block graphs """
start_time_hybridblock = time.time()
hybrid_block_graph, label_community_block, label_block_community = main.multilevel_algo.create_hybridblockgraph_from_hybridtractgraph(hybrid_tract_graph, block_graph, label_community_tract, label_tract_community,which_level='block')
hybrid_blockgroup_graph, label_community_blockgroup, label_blockgroup_community = main.multilevel_algo.create_hybridblockgraph_from_hybridtractgraph(hybrid_tract_graph, blockgroup_graph, label_community_tract, label_tract_community,which_level='blockgroup')
print(len(hybrid_tract_graph),len(hybrid_blockgroup_graph),len(hybrid_block_graph))
print(time.time()-start_time_hybridblock,"secs to read the datafiles")



""" Algorithm parameters"""
n_iterations_no_improvement = 1000

trackmap_filename = "results/curated_maps/MapA/mapA_stage3_tract"
# trackmap_filename = "results/curated_maps/MapB/mapB_stage3_tract"
# trackmap_filename = "results/curated_maps/MapC/mapC_stage3_tract"

do_popbal_localsearch(trackmap_filename)


# blockmap_filename = "results/curated_maps/MapA/mapA_stage4_block"
# blockmap_filename = "results/curated_maps/MapB/mapB_stage4_block"
# blockmap_filename = "results/curated_maps/MapC/mapC_stage4_block"
# read_blockmap_save_analytics(blockmap_filename)
