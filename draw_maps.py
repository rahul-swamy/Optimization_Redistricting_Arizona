
# Optimap: A Practical Optimization Framework for Congressional Redistricting in Arizona


# Copyright © 2022 Institute for Computational Redistricting (ICOR), University of Illinois at Urbana-Champaign


import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from heapq import nlargest
import math
import operator
from functools import reduce
import gerrychain
import pandas as pd
from collections import defaultdict


class input:
    def read_data_county_2020():
        G = nx.Graph()

        with open('datasets/04_county_adjacency_miles.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            rowlist = [row for row in spamreader]
            edgelist = []
            bndry = {}
            for i in range(len(rowlist)):
                edgelist.append((int(rowlist[i][0]),int(rowlist[i][1])))
                bndry[int(rowlist[i][0]),int(rowlist[i][1])] = float(rowlist[i][2])

        G.add_edges_from(edgelist)

        for (i,j) in bndry:
            G[i][j]['bndry'] = bndry[(i,j)]

        for i in G.nodes():
            G.nodes[i]['p_dem'] = 0
            G.nodes[i]['p_rep'] = 0
            G.nodes[i]['pop'] = 0

        with open('datasets/04_county_coords.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_list = [row for row in spamreader]
            x_coord, y_coord = {}, {}
            for i in range(len(row_list)):
                x_coord[int(row_list[i][0])] = float(row_list[i][1])
                y_coord[int(row_list[i][0])] = float(row_list[i][2])

        for i in G.nodes():
            G.nodes[i]['x'] = x_coord[i]
            G.nodes[i]['y'] = y_coord[i]

        return G


    def read_data_tract_2020():
        G = nx.Graph()

        with open('datasets/04_tract_adjacency_miles.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            rowlist = [row for row in spamreader]
            edgelist = []
            bndry = {}
            for i in range(len(rowlist)):
                if int(rowlist[i][0]) != int(rowlist[i][1]):
                    edgelist.append((int(rowlist[i][0]),int(rowlist[i][1])))
                    bndry[int(rowlist[i][0]),int(rowlist[i][1])] = float(rowlist[i][2])

        G.add_edges_from(edgelist)
        for (i,j) in bndry:
            G[i][j]['bndry'] = bndry[(i,j)]

        with open('datasets/2020_tracts_allstates_hispanicornot_allraces.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_list = [row for row in spamreader]
            pop = {}
            p_maj_race, p_min_race, p_hispanic, p_black, p_asian, p_amindian, p_pacific = {}, {}, {}, {}, {}, {}, {}
            for i in range(1,len(row_list)):
                tract = int(row_list[i][0])
                if i != 0 and int(tract/1000000000) == 4:
                    pop[tract] = int(float(row_list[i][1]))
                    p_maj_race[tract] = int(float(row_list[i][2]))
                    p_min_race[tract] = int(float(row_list[i][9]))
                    p_hispanic[tract] = int(float(row_list[i][3]))
                    p_black[tract] = int(float(row_list[i][4]))
                    p_amindian[tract] = int(float(row_list[i][5]))
                    p_asian[tract] = int(float(row_list[i][6]))
                    p_pacific[tract] = int(float(row_list[i][7]))

        voting_data = pd.read_csv('datasets/AZ_tract_votingdata_9_elections_2018-20_all.csv', sep=',', delimiter=None, header = 0)#, usecols = ['combined_fips','votes_dem','votes_gop'],names = ['county_FIPS','votes_dem','votes_gop'],dtype = float)#, nrows = 100)
        voting_data_dict = voting_data.T.to_dict()

        elections = ['G18USSDSIN', 'G18GOVDGAR', 'G18SOSDHOB', 'G18ATGDCON', 'G18TREDMAN', 'G18SPIDHOF', 'G18MNIDPIE', 'G20PREDBID','G20USSDKEL']
        elections += ['G18USSRMCS', 'G18GOVRDUC', 'G18SOSRGAY', 'G18ATGRBRN', 'G18TRERYEE', 'G18SPIRRIG', 'G18MNIRHAR', 'G20PRERTRU','G20USSRMCS']

        p_dem, p_rep = {}, {}
        voters_dict = {election: {} for election in elections}
        for key, val in voting_data_dict.items():
            tract = int(val['tract'])
            if int(tract/1000000000) == 4 and tract in G.nodes():
                p_dem[tract] = val['p_dem']
                p_rep[tract] = val['p_rep']
                for election in elections:
                    voters_dict[election][tract] = val[election]

        with open('datasets/04_tract_coords.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            row_list = [row for row in spamreader]
            x_coord, y_coord = {}, {}
            for i in range(len(row_list)):
                x_coord[int(row_list[i][0])] = float(row_list[i][1])
                y_coord[int(row_list[i][0])] = float(row_list[i][2])

        # print(len(G.nodes()), "tracts in tract graph")
        for i in G.nodes():
            G.nodes[i]['x'] = x_coord[i]
            G.nodes[i]['y'] = y_coord[i]
            G.nodes[i]['pop'] = pop[i]
            G.nodes[i]['p_dem'] = p_dem[i]
            G.nodes[i]['p_rep'] = p_rep[i]
            G.nodes[i]['p_maj'] = p_maj_race[i]
            G.nodes[i]['p_min'] = p_min_race[i]
            G.nodes[i]['p_hispanic'] = p_hispanic[i]
            G.nodes[i]['p_black'] = p_black[i]
            G.nodes[i]['p_amindian'] = p_amindian[i]
            G.nodes[i]['p_asian'] = p_asian[i]
            G.nodes[i]['p_pacific'] = p_pacific[i]
            for election in elections:
                G.nodes[i][election] = voters_dict[election][i]

        # total_white = sum(G.nodes[i]['p_maj'] for i in G.nodes())
        # total_min = sum(G.nodes[i]['p_min'] for i in G.nodes())
        # print("Fraction of minorities in AZ:", total_min/(total_min+total_white))

        return G


    def read_communities_of_interest_and_places():
        tracts_coi, tracts_place = {}, {}
        with open('datasets/tracts_preservedinAZCOI.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            rowlist = [row for row in spamreader]
            edgelist = []
            bndry = {}
            for i in range(1,len(rowlist)):
                if int(rowlist[i][0]) not in tracts_coi:
                    tracts_coi[int(rowlist[i][0])] = [int(rowlist[i][2])]
                else:
                    tracts_coi[int(rowlist[i][0])].append(int(rowlist[i][2]))


        with open('datasets/tracts_preservedinAZplaces_attributes.csv') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            rowlist = [row for row in spamreader]
            edgelist = []
            bndry = {}
            for i in range(1,len(rowlist)):
                if int(rowlist[i][0]) not in tracts_place:
                    tracts_place[int(rowlist[i][0])] = [int(rowlist[i][1])]
                else:
                    tracts_place[int(rowlist[i][0])].append(int(rowlist[i][1]))

        return tracts_coi, tracts_place


class multilevel_algo:

    def solve_matching_greedy(input_graph):
        graph_copy = nx.Graph()
        graph_copy.add_edges_from([(i,j) for (i,j) in input_graph.edges()])
        for (i,j) in graph_copy.edges():
            graph_copy[i][j]['weight'] = input_graph[i][j]['weight']
        matching_edges_list = []
        iterations_so_far = 0
        while(len(graph_copy.edges()) > 0):
            edge_weight_dict = {(i,j): graph_copy[i][j]['weight'] for (i,j) in graph_copy.edges()}
            best_edge = min(iter(edge_weight_dict.items()), key=operator.itemgetter(1))[0]
            matching_edges_list.append(best_edge)

            graph_copy.remove_nodes_from([best_edge[0],best_edge[1]])
            iterations_so_far += 1
        return matching_edges_list


    def create_hybrid_graph(county_graph, tract_graph, n_counties_to_split, communities_dict, places_dict, which_level='tract'):
        elections = ['G18USSDSIN', 'G18GOVDGAR', 'G18SOSDHOB', 'G18ATGDCON', 'G18TREDMAN', 'G18SPIDHOF', 'G18MNIDPIE', 'G20PREDBID','G20USSDKEL']
        elections += ['G18USSRMCS', 'G18GOVRDUC', 'G18SOSRGAY', 'G18ATGRBRN', 'G18TRERYEE', 'G18SPIRRIG', 'G18MNIRHAR', 'G20PRERTRU','G20USSRMCS']

        """ Assigning majority and minority voters and voting data from tracts to counties """
        for i in county_graph.nodes():
            county_graph.nodes[i]['pop'] = 0

        county_id = {}
        for i in tract_graph.nodes():
            if which_level == 'tract':
                county_id[i] = int(i/1000000)
            else:
                county_id[i] = int(i/10000000000)
            county_graph.nodes[county_id[i]]['pop'] += tract_graph.nodes[i]['pop']

        """ Split most populated counties """
        pop_dict = {i: county_graph.nodes[i]['pop'] for i in county_graph.nodes()}
        most_pop_counties_to_break = nlargest(n_counties_to_split, pop_dict, key=pop_dict.get)
        print("Counties broken:",sorted(most_pop_counties_to_break))
        tracts_in_broken_counties = []
        for county in most_pop_counties_to_break:
            tracts_in_broken_counties += [i for i in tract_graph.nodes() if county_id[i] == county]

        preserved_counties = list(set(county_graph.nodes()) - set(most_pop_counties_to_break))
        tracts_in_preserved_counties = []
        tracts_in_preserved_counties_dict = {}
        for county in preserved_counties:
            tracts_in_preserved_counties += [i for i in tract_graph.nodes() if county_id[i] == county]
            tracts_in_preserved_counties_dict[county] = [i for i in tract_graph.nodes() if county_id[i] == county]

        """ Create communities of interest, preserved places and preserved counties """
        label_community_tract, label_tract_community = {}, {}
        COI_index = 1
        tracts_in_communities = set()
        for c in communities_dict:
            if len(communities_dict[c]) > 1:
                tracts_in_communities = tracts_in_communities.union(set(communities_dict[c]))
                label_community_tract[COI_index] = communities_dict[c]
                for i in communities_dict[c]:
                    label_tract_community[i] = COI_index
                COI_index += 1

        for c in places_dict:
            if len(places_dict[c]) > 1:
                tracts_in_communities = tracts_in_communities.union(set(places_dict[c]))
                label_community_tract[COI_index] = places_dict[c]
                for i in places_dict[c]:
                    label_tract_community[i] = COI_index
                COI_index += 1


        for county in preserved_counties:
            tracts_in_communities = tracts_in_communities.union(set(tracts_in_preserved_counties_dict[county]))
            label_community_tract[COI_index] = tracts_in_preserved_counties_dict[county]
            for i in tracts_in_preserved_counties_dict[county]:
                label_tract_community[i] = COI_index
            COI_index += 1

        print("Total number of tracts in communities & preserved counties:",len(tracts_in_communities))
        communities = set([label_tract_community[i] for i in tracts_in_communities])
        unpreserved_tracts = set(tract_graph.nodes())-set(tracts_in_communities)
        print("Total number of communities:",len(communities))

        """ Create new non-overlapping communities """
        comcom_overlap_graph = nx.Graph() # Graph of overlapping communities; two communities share an edge if they overlap
        comcom_overlap_graph.add_nodes_from(communities)
        for c1 in communities:
            for c2 in communities:
                n_tracts_overlapping = len(set(label_community_tract[c1]).intersection(set(label_community_tract[c2])))
                if c1 < c2 and n_tracts_overlapping > 0:
                    comcom_overlap_graph.add_edge(c1,c2)

        print("Pair-wise overlapping communities:",comcom_overlap_graph.edges())
        label_community_tract_updated, label_tract_community_updated = {}, {}
        comm_index = 0
        components = nx.connected_components(comcom_overlap_graph)
        for component in components:
            comm_index += 1
            tracts_in_comm = []
            for c in component:
                tracts_in_comm += label_community_tract[c]
                for i in label_community_tract[c]:
                    label_tract_community_updated[i] = comm_index
            label_community_tract_updated[comm_index] = set(tracts_in_comm)

        """ Reset the overlapping communities into non-overlapping ones """
        communities = set(label_community_tract_updated.keys())
        label_community_tract = label_community_tract_updated.copy()
        label_tract_community = label_tract_community_updated.copy()
        print("Total number of non-overlapping preserved regions:",len(communities))

        """ Create hybrid graph """
        hybrid_graph = tract_graph.copy()
        hybrid_graph.remove_nodes_from(tracts_in_communities)
        hybrid_graph.add_nodes_from(communities)

        for c in communities:
            hybrid_graph.nodes[c]['pop'] = sum(tract_graph.nodes[i]['pop'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_maj'] = sum(tract_graph.nodes[i]['p_maj'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_min'] = sum(tract_graph.nodes[i]['p_min'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_rep'] = sum(tract_graph.nodes[i]['p_rep'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_dem'] = sum(tract_graph.nodes[i]['p_dem'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_hispanic'] = sum(tract_graph.nodes[i]['p_hispanic'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_black'] = sum(tract_graph.nodes[i]['p_black'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_amindian'] = sum(tract_graph.nodes[i]['p_amindian'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_asian'] = sum(tract_graph.nodes[i]['p_asian'] for i in label_community_tract[c])
            hybrid_graph.nodes[c]['p_pacific'] = sum(tract_graph.nodes[i]['p_pacific'] for i in label_community_tract[c])

            for election in elections:
                hybrid_graph.nodes[c][election] = sum(tract_graph.nodes[i][election] for i in label_community_tract[c])

            hybrid_graph.nodes[c]['x'] = sum(tract_graph.nodes[i]['x'] for i in label_community_tract[c])/len(label_community_tract[c])
            hybrid_graph.nodes[c]['y'] = sum(tract_graph.nodes[i]['y'] for i in label_community_tract[c])/len(label_community_tract[c])

        for (i,j) in tract_graph.edges():
            if i in tracts_in_communities:
                c_i = label_tract_community[i]
                if j in tracts_in_communities:  # i and j are in communities: we wil lad a community-community edge
                    c_j = label_tract_community[j]
                    if c_i != c_j:
                        if (c_i,c_j) not in hybrid_graph.edges():
                            hybrid_graph.add_edge(c_i,c_j)
                            hybrid_graph[c_i][c_j]['bndry'] = tract_graph[i][j]['bndry']
                        else:
                            hybrid_graph[c_i][c_j]['bndry'] += tract_graph[i][j]['bndry']
                else: # i is in a community, j is not:  we will add a community-tract edge
                    if (c_i,j) not in hybrid_graph.edges():
                        hybrid_graph.add_edge(c_i,j)
                        hybrid_graph[c_i][j]['bndry'] = tract_graph[i][j]['bndry']
                    else:
                        hybrid_graph[c_i][j]['bndry'] += tract_graph[i][j]['bndry']
            else:
                if j in tracts_in_communities: # i is not in a community, j is in a community: we will add a community-tract edge
                    c_j = label_tract_community[j]
                    if (i,c_j) not in hybrid_graph.edges():
                        hybrid_graph.add_edge(i,c_j)
                        hybrid_graph[i][c_j]['bndry'] = tract_graph[i][j]['bndry']
                    else:
                        hybrid_graph[i][c_j]['bndry'] += tract_graph[i][j]['bndry']

        return hybrid_graph, label_community_tract, label_tract_community


    def create_hybridblockgraph_from_hybridtractgraph(hybrid_tract_graph, block_graph, label_community_tract, label_tract_community, which_level='block'):
        # this function creates a hybrid graph, a graph with a mixture of communites and census blocks (or block groups)
        hybrid_block_graph = block_graph.copy()
        print(list(block_graph.nodes())[0])

        """ Collect the blocks in communities """
        blocks_in_communities, label_community_block, label_block_community = [], {}, {}
        communities = [i for i in hybrid_tract_graph.nodes() if i < 9999]
        print("Communities:",communities)
        tract_id = {}
        for i in hybrid_block_graph.nodes():
            tract_id[i] = int(i/10000) if which_level == 'block' else int(i/10)
            if tract_id[i] in label_tract_community:
                community_id = label_tract_community[tract_id[i]]
                blocks_in_communities.append(i)
                label_community_block.setdefault(community_id, []).append(i)
                label_block_community[i] = community_id

        print(len(blocks_in_communities),which_level+"s in communities")

        hybrid_block_graph = nx.read_gpickle('datasets/hybrid_'+which_level+'_graph.gpickle')

        """ The following commented code creates a hybrid block graph.
        The data read from the gpickle code above was obtained from the below commented code.
        Given the computational time taken for the below code, we only need to read from the gpickle file henceforth. """

        # """ Remove the blocks in communities from hybrid graph """
        # hybrid_block_graph.remove_nodes_from(blocks_in_communities)
        #
        # """ Add the communities to the hybrid graph """
        # hybrid_block_graph.add_nodes_from(communities)
        # node_attributes = dict(hybrid_tract_graph.nodes(data=True))
        #
        # nx.set_node_attributes(hybrid_block_graph, node_attributes)
        #
        # """ Add edges between communities """
        # comcom_edges = [(i,j) for (i,j) in hybrid_tract_graph.edges() if i in communities and j in communities]
        # hybrid_block_graph.add_edges_from(comcom_edges)
        # for c_i,c_j in comcom_edges:
        #     hybrid_block_graph[c_i][c_j]['bndry'] = hybrid_tract_graph[c_i][c_j]['bndry']
        #
        # tracts_that_neighbor_communities = [i for (i,j) in hybrid_tract_graph.edges() if j in communities]
        # blocks_that_neighbor_communities = [i for i in block_graph.nodes() if tract_id[i] in tracts_that_neighbor_communities and i not in blocks_in_communities]
        # print(len(blocks_that_neighbor_communities),"blocks neighbor communities")
        # """ Add edges between communities and blocks """
        # for i in blocks_that_neighbor_communities:
        #     for j in block_graph.neighbors(i):
        #         if j in blocks_in_communities:
        #             if (i,label_block_community[j]) not in hybrid_block_graph.edges():
        #                 hybrid_block_graph.add_edge(i,label_block_community[j])
        #                 if (i,j) in block_graph.edges():
        #                     hybrid_block_graph[i][label_block_community[j]]['bndry'] = block_graph[i][j]['bndry']
        #                 else:
        #                     hybrid_block_graph[i][label_block_community[j]]['bndry'] = block_graph[j][i]['bndry']
        #             else:
        #                 if (i,j) in block_graph.edges():
        #                     hybrid_block_graph[i][label_block_community[j]]['bndry'] += block_graph[i][j]['bndry']
        #                 else:
        #                     hybrid_block_graph[i][label_block_community[j]]['bndry'] += block_graph[j][i]['bndry']
        #
        # print(len(hybrid_block_graph.nodes),"nodes,",len(hybrid_block_graph.edges()),"edges in the hybrid",which_level,"graph")
        #
        # nx.write_gpickle(hybrid_block_graph, "hybrid_"+which_level+"_graph.gpickle")

        return hybrid_block_graph, label_community_block, label_block_community


    def aggregate_multilevel(tract_graph, hybrid_graph, label_community_tract, n_levels, greedy_coarsening_or_not=1, plot_coarse_levels=False):
        level_graph = {}
        level_l_to_l_plus_1 = {}
        level_l_plus_1_to_l = {}
        level_graph[0] = hybrid_graph.copy()

        print("Number of edges in level 0 graph:",len(level_graph[0].edges()), "Number of nodes:",len(level_graph[0].nodes()))

        """ AGGREGATING FROM LELEL 0 """
        for l in range(1,n_levels+1):
            pop_dict = {i: level_graph[l-1].nodes[i]['pop'] for i in level_graph[l-1].nodes()}
            M = max(pop_dict[i] for i in pop_dict)
            print("population range in level",l-1,"is",min(pop_dict[i] for i in pop_dict),"-",max(pop_dict[i] for i in pop_dict),"average:",float(sum(pop_dict[i] for i in pop_dict))/len(pop_dict))

            edge_weights = {(i,j): (2*M - (pop_dict[i]+pop_dict[j])) for (i,j) in level_graph[l-1].edges()}
            M_min = min(edge_weights[i] for i in edge_weights)
            M_max = max(edge_weights[i] for i in edge_weights)
            edge_weights = {(i,j): float(float(edge_weights[i,j] - M_min)/float(M_max)) for (i,j) in edge_weights}

            for (i,j) in level_graph[l-1].edges():
                level_graph[l-1][i][j]['weight'] = (edge_weights[i,j])

            matchings = multilevel_algo.solve_matching_greedy(level_graph[l-1])

            print("Number of matched edges:",len(matchings))

            matched_nodes = [i for i in level_graph[l-1].nodes() if any(i in edge for edge in matchings)]
            unmatched_nodes = [i for i in level_graph[l-1].nodes() if i not in matched_nodes]

            level_graph[l] = nx.Graph()
            level_l_to_l_plus_1[l-1] = {}
            level_l_plus_1_to_l[l] = {}
            node_counter = 0
            pop_count = 0
            for (i,j) in matchings:
                level_l_to_l_plus_1[l-1][i] = node_counter
                level_l_to_l_plus_1[l-1][j] = node_counter
                level_l_plus_1_to_l[l][node_counter] = [i,j]

                level_graph[l].add_node(node_counter, pop = pop_dict[i] + pop_dict[j], p_dem = level_graph[l-1].nodes[i]['p_dem'] + level_graph[l-1].nodes[j]['p_dem'], p_rep = level_graph[l-1].nodes[i]['p_rep'] + level_graph[l-1].nodes[j]['p_rep'], p_maj = level_graph[l-1].nodes[i]['p_maj'] + level_graph[l-1].nodes[j]['p_maj'], p_min = level_graph[l-1].nodes[i]['p_min'] + level_graph[l-1].nodes[j]['p_min'], \
                                    x =  float(level_graph[l-1].nodes[i]['x']+level_graph[l-1].nodes[j]['x'])/2, y =  float(level_graph[l-1].nodes[i]['y']+level_graph[l-1].nodes[j]['y'])/2)
                pop_count += level_graph[l-1].nodes[i]['pop'] + level_graph[l-1].nodes[j]['pop']
                level_graph[l].nodes[node_counter]['pop'] = level_graph[l-1].nodes[i]['pop'] + level_graph[l-1].nodes[j]['pop']
                level_graph[l].nodes[node_counter]['p_hispanic'] = level_graph[l-1].nodes[i]['p_hispanic'] + level_graph[l-1].nodes[j]['p_hispanic']
                level_graph[l].nodes[node_counter]['p_black'] = level_graph[l-1].nodes[i]['p_black'] + level_graph[l-1].nodes[j]['p_black']
                level_graph[l].nodes[node_counter]['p_amindian'] = level_graph[l-1].nodes[i]['p_amindian'] + level_graph[l-1].nodes[j]['p_amindian']
                level_graph[l].nodes[node_counter]['p_asian'] = level_graph[l-1].nodes[i]['p_asian'] + level_graph[l-1].nodes[j]['p_asian']
                level_graph[l].nodes[node_counter]['p_pacific'] = level_graph[l-1].nodes[i]['p_pacific'] + level_graph[l-1].nodes[j]['p_pacific']
                node_counter += 1

            for i in unmatched_nodes:
                pop_count += level_graph[l-1].nodes[i]['pop']
                level_l_to_l_plus_1[l-1][i] = node_counter
                level_l_plus_1_to_l[l][node_counter] = [i]

                level_graph[l].add_node(node_counter, pop = pop_dict[i], p_dem = level_graph[l-1].nodes[i]['p_dem'], p_rep = level_graph[l-1].nodes[i]['p_rep'], p_maj = level_graph[l-1].nodes[i]['p_maj'], p_min = level_graph[l-1].nodes[i]['p_min'],
                                                        x = level_graph[l-1].nodes[i]['x'], y = level_graph[l-1].nodes[i]['y'])

                level_graph[l].nodes[node_counter]['pop'] = level_graph[l-1].nodes[i]['pop']
                level_graph[l].nodes[node_counter]['p_hispanic'] = level_graph[l-1].nodes[i]['p_hispanic']
                level_graph[l].nodes[node_counter]['p_black'] = level_graph[l-1].nodes[i]['p_black']
                level_graph[l].nodes[node_counter]['p_amindian'] = level_graph[l-1].nodes[i]['p_amindian']
                level_graph[l].nodes[node_counter]['p_asian'] = level_graph[l-1].nodes[i]['p_asian']
                level_graph[l].nodes[node_counter]['p_pacific'] = level_graph[l-1].nodes[i]['p_pacific']
                node_counter += 1

            """ Tally the boundary lengths for the segments in the coarse graph """
            for (i,j) in level_graph[l-1].edges():
                upper_level_unit_of_i = level_l_to_l_plus_1[l-1][i]
                upper_level_unit_of_j = level_l_to_l_plus_1[l-1][j]
                if upper_level_unit_of_i != upper_level_unit_of_j:
                    level_graph[l].add_edge(upper_level_unit_of_i, upper_level_unit_of_j)
                    level_graph[l][upper_level_unit_of_i][upper_level_unit_of_j]['bndry'] = 0

            for (i,j) in level_graph[l-1].edges():
                upper_level_unit_of_i = level_l_to_l_plus_1[l-1][i]
                upper_level_unit_of_j = level_l_to_l_plus_1[l-1][j]
                if upper_level_unit_of_i != upper_level_unit_of_j:
                    level_graph[l][upper_level_unit_of_i][upper_level_unit_of_j]['bndry'] += level_graph[l-1][i][j]['bndry']

            print("\nNumber of nodes in level", l ,"graph:",len(level_graph[l].nodes()))
            print("pop_max:", max([level_graph[l].nodes[i]['pop'] for i in level_graph[l]]), "pop_min:", min([level_graph[l].nodes[i]['pop'] for i in level_graph[l]]), "ratio:", float(max([level_graph[l].nodes[i]['pop'] for i in level_graph[l]]))/(min([level_graph[l].nodes[i]['pop'] for i in level_graph[l]])+1))
            print("pop_avg:", float(sum([level_graph[l].nodes[i]['pop'] for i in level_graph[l]]))/len(level_graph[l]), "ratio:", float(max([level_graph[l].nodes[i]['pop'] for i in level_graph[l]]))/(float(sum([level_graph[l].nodes[i]['pop'] for i in level_graph[l]])+1)/len(level_graph[l])))
            print("\n")
            # print("Number of majority voters:", sum([level_graph[l].nodes[i]['p_maj'] for i in level_graph[l]]), "Number of minority voters:", sum([level_graph[l].nodes[i]['p_min'] for i in level_graph[l]]))

            """ Plot the coarse levels """
            if plot_coarse_levels == True:
                print("Level",l,"graph:")
                """ Unwrap the l-th level to the 0-th level"""
                new_dict = {l_prime: {} for l_prime in range(1,l+1)}
                for i in level_graph[l].nodes():
                    """ Initialize """
                    new_dict[l][i] = []
                    for j_prime in level_l_plus_1_to_l[l][i]:
                        if l == 1 and j_prime < 100:
                            new_dict[l][i] += label_community_tract[j_prime]
                        else:
                            new_dict[l][i].append(j_prime)

                    """ Recurse through the levels down to 0 """
                    for l_prime in sorted(list(range(1,l)),reverse = 1):
                        new_dict[l_prime][i] = []
                        for j in new_dict[l_prime+1][i]:
                            for j_prime in level_l_plus_1_to_l[l_prime][j]:
                                if l_prime == 1 and j_prime < 100:
                                    new_dict[l_prime][i] += label_community_tract[j_prime] # tracts in community j_prime
                                else:
                                    new_dict[l_prime][i].append(j_prime)


        return level_graph[n_levels], level_graph[0], level_l_to_l_plus_1, level_l_plus_1_to_l, level_graph


class metric:
    def compute_partisan_symmetry(z_k, graph,plot_or_not):
        total_voted = sum(graph.nodes[i]['p_dem'] + graph.nodes[i]['p_rep'] for i in graph)
        curr_net_vote_share = float(sum(graph.nodes[i]['p_dem']for i in graph))/total_voted
        curr_net_seat_share = float(len([k for k in z_k if sum(graph.nodes[i]['p_dem']for i in z_k[k]) >= sum(graph.nodes[i]['p_rep']for i in z_k[k])]))/K
        original_net_vote_share = curr_net_vote_share
        original_net_seat_share = curr_net_seat_share
        voted_k = {k: sum(graph.nodes[i]['p_dem'] + graph.nodes[i]['p_rep'] for i in z_k[k]) for k in range(1,K+1)}
        pdem_levels_k = {k: float(sum(graph.nodes[i]['p_dem'] for i in z_k[k]))/sum(graph.nodes[i]['p_dem'] + graph.nodes[i]['p_rep'] for i in z_k[k]) for k in range(1,K+1)}
        original_pdem_levels_k = pdem_levels_k.copy()

        voteshare_list = [0,1]
        seatsare_list = [0,1]
        seat_share_dict = {0:0, 1:1}
        pdem_levels_lessthanhalf_k = {k: pdem_levels_k[k] for k in pdem_levels_k if pdem_levels_k[k] < 0.5}
        while (len(pdem_levels_lessthanhalf_k) != 0):
            # print "< half:",pdem_levels_lessthanhalf_k
            next_k = max(iter(pdem_levels_lessthanhalf_k.items()), key=operator.itemgetter(1))[0]
            fraction_incremented = float(0.5 - pdem_levels_lessthanhalf_k[next_k])
            pdem_levels_k = {k: min(pdem_levels_k[k] + fraction_incremented,1) for k in range(1,K+1)}
            curr_net_vote_share = float(sum(pdem_levels_k[k]*voted_k[k] for k in range(1,K+1)))/total_voted
            curr_net_seat_share += float(1)/K
            seat_share_dict[curr_net_vote_share] = curr_net_seat_share
            seatsare_list.append(curr_net_seat_share)
            voteshare_list.append(curr_net_vote_share)
            pdem_levels_lessthanhalf_k = {k: pdem_levels_lessthanhalf_k[k] + fraction_incremented for k in pdem_levels_lessthanhalf_k}
            pdem_levels_lessthanhalf_k.pop(next_k)


        pdem_levels_k = original_pdem_levels_k.copy()
        curr_net_vote_share = original_net_vote_share
        curr_net_seat_share = original_net_seat_share
        pdem_levels_morethan_k = {k: pdem_levels_k[k] for k in pdem_levels_k if pdem_levels_k[k] >= 0.5}
        while (len(pdem_levels_morethan_k) != 0):
            next_k = min(iter(pdem_levels_morethan_k.items()), key=operator.itemgetter(1))[0]
            fraction_incremented = float(pdem_levels_morethan_k[next_k] - 0.5)
            pdem_levels_k = {k: max(pdem_levels_k[k] - fraction_incremented, 0) for k in range(1,K+1)}
            # print pdem_levels_k
            curr_net_vote_share = float(sum(pdem_levels_k[k]*voted_k[k] for k in range(1,K+1)))/total_voted
            seatsare_list.append(curr_net_seat_share)
            seat_share_dict[curr_net_vote_share] = curr_net_seat_share
            curr_net_seat_share -= float(1)/K
            voteshare_list.append(curr_net_vote_share)
            pdem_levels_morethan_k = {k: pdem_levels_morethan_k[k] - fraction_incremented for k in pdem_levels_morethan_k}
            pdem_levels_morethan_k.pop(next_k)

        voteshare_list = sorted(voteshare_list)
        seatsare_list = sorted(seatsare_list)
        flipped_voteshare_list = [1 - i for i in voteshare_list]
        flipped_seatsare_list = [1-i for i in seatsare_list]


        """ Calculating the area now """

        flipped_seat_share_dict = {1-i: 1-seat_share_dict[i]+float(1)/K for i in seat_share_dict}
        prev_breakpoint_fn = 0
        prev_breakpoint_flipped_fn = 0
        prev_fn_val = 0
        prev_flippedfn_val = 0
        seat_share_dict.pop(0)
        flipped_seat_share_dict.pop(0)
        curr_breakpoint = 0
        next_breakpoint = min(min(seat_share_dict),min(flipped_seat_share_dict))
        area = 0
        while(1):
            curr_breakpoint = next_breakpoint
            if curr_breakpoint in seat_share_dict:
                prev_breakpoint_fn = curr_breakpoint
                prev_fn_val = seat_share_dict[prev_breakpoint_fn]
                seat_share_dict.pop(prev_breakpoint_fn)
            else:
                prev_breakpoint_flipped_fn = curr_breakpoint
                prev_flippedfn_val = flipped_seat_share_dict[prev_breakpoint_flipped_fn]
                flipped_seat_share_dict.pop(prev_breakpoint_flipped_fn)

            if len(flipped_seat_share_dict)==0 and len(seat_share_dict)>0:
                next_breakpoint = min(seat_share_dict)
            elif len(flipped_seat_share_dict)>0 and len(seat_share_dict)==0:
                next_breakpoint = min(flipped_seat_share_dict)
            elif len(flipped_seat_share_dict)>0 and len(seat_share_dict)>0:
                next_breakpoint = min(min(seat_share_dict),min(flipped_seat_share_dict))
            else:
                break
            area += (next_breakpoint - curr_breakpoint)*abs(prev_fn_val-prev_flippedfn_val)

        if plot_or_not:
            print("partisan asymmetry:", area)
            ax = plt.figure().gca()
            ax.set_xticks(np.arange(0, 1, 0.1))
            ax.set_yticks(np.arange(0, 1., 0.1))
            plt.grid()
            demplot, = plt.step(voteshare_list, seatsare_list, where = 'post', color = 'b', label = 'Democrats')
            repplot, = plt.step(flipped_voteshare_list, flipped_seatsare_list, where = 'post', color = 'r', linestyle='--', label = 'Republicans')
            plt.axes()
            plt.xlim([0, 1])
            plt.xlabel('Average vote share across all districts', fontsize = 20)
            plt.ylabel('Seat share (fraction of districts won)', fontsize = 20)
            plt.legend(handles = [demplot, repplot], loc=2)
            plt.show()

        return area


    def compute_partisan_symmetry_with_formula(z_k, graph):
        K = len(z_k)
        u_k_dict = {k: float(sum(graph.nodes[i]['p_rep'] for i in z_k[k]))/sum(graph.nodes[i]['p_rep']+graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
        u_k_list = sorted([u_k_dict[k] for k in u_k_dict], reverse = 1)

        v_j = {j: float(sum(abs(min(1,max(0, u_k_dict[k] - u_k_list[j-1] + 0.5))) for k in z_k))/K for j in z_k}

        PA = float(sum(abs(v_j[j]+v_j[K-j+1] - 1) for j in z_k))/K

        return PA


    def evaluate_effgap(z_k, graph):
        z_k_copy = z_k.copy()

        total_voted = 0
        net_wasted_votes = 0
        for k in z_k:
            rep_pop = sum(graph.nodes[i]['p_rep'] for i in z_k[k])
            dem_pop = sum(graph.nodes[i]['p_dem'] for i in z_k[k])
            voted_pop = sum(graph.nodes[i]['p_rep']+graph.nodes[i]['p_dem'] for i in z_k[k])

            if rep_pop > dem_pop:
                who_won = 'R'
                dem_wasted_votes = dem_pop
                rep_wasted_votes = rep_pop - voted_pop*0.5
            else:
                who_won = 'D'
                dem_wasted_votes = dem_pop - voted_pop*0.5
                rep_wasted_votes = rep_pop
            total_voted += voted_pop
            net_wasted_votes += dem_wasted_votes-rep_wasted_votes
        eff_gap = float(net_wasted_votes)/float(total_voted)
        return eff_gap


    def evaluate_number_of_cmptitv_dists(z_k, graph, margin_of_victory):
        pop_rep_k = {k: sum(graph.nodes[i]['p_rep'] for i in z_k[k]) for k in z_k}
        pop_dem_k = {k: sum(graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
        pop_voted_k = {k: pop_rep_k[k]+pop_dem_k[k]  for k in z_k}
        return [k for k in z_k if float(abs(pop_rep_k[k]-pop_dem_k[k]))/pop_voted_k[k] <= margin_of_victory], [float(pop_rep_k[k]-pop_dem_k[k])/pop_voted_k[k] for k in z_k]


    def evaluate_max_margin(z_k, graph):
        pop_rep_k = {k: sum(graph.nodes[i]['p_rep'] for i in z_k[k]) for k in z_k}
        pop_dem_k = {k: sum(graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
        return max([float(abs(pop_rep_k[k]-pop_dem_k[k]))/(pop_rep_k[k]+pop_dem_k[k]) for k in z_k])


    def compute_euclidean_distance(graph):
        length = {i: {j: 0 for j in graph.nodes()} for i in graph.nodes()}
        length = {}
        for i in graph.nodes():
            length[i] = {}
            x_i = graph.nodes()[i]['x']
            y_i = graph.nodes()[i]['y']
            for j in graph.nodes():
                x_j = graph.nodes()[j]['x']
                y_j = graph.nodes()[j]['y']
                length[i][j] = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
        return length


    def evaluate_compactness_sumdist(z_k, graph, d_matrix):
        district_centers_k = {}
        for k in range(1,K+1):
            distance_dict_i = {i: sum(graph.nodes[j]['pop']*d_matrix[i][j]**2 for j in z_k[k]) for i in z_k[k]}

            district_centers_k[k] = min(list(distance_dict_i.items()), key=lambda x: x[1])[0]

        return sum(sum(graph.nodes[j]['pop']*d_matrix[district_centers_k[k]][j]**2 for j in z_k[k]) for k in z_k), district_centers_k



    def contiguity_check(z_k,graph):
        for k in z_k:
            sub_graph = nx.Graph()
            sub_graph = graph.subgraph(z_k[k])
            if not nx.is_connected(sub_graph):
                return k,0
        return K, 1


    def check_potential_contiguity(z_k_yes, z_k_no):
        flag = 0
        for k in range(1,K+1):
            sub_graph = nx.Graph()
            sub_graph.add_edges_from(planar_graph.edges())
            sub_graph.remove_nodes_from(z_k_no[k])
            sub_graph.remove_nodes_from([i for k1 in range(1,K+1) for i in z_k_yes[k1] if k1!=k])
            if not nx.is_connected(sub_graph):
                return 1

        return flag


    def evaluate_compactness_edgecuts(z_i, graph):
        weight_of_cuts = 0
        for (i,j) in graph.edges():
            if z_i[i] != z_i[j]:
                weight_of_cuts += graph[i][j]['bndry']
        return weight_of_cuts


    def evaluate_population_balance(z_k,graph):
        pop_k = {k: sum(graph.nodes[i]['pop'] for i in z_k[k]) for k in range(1,K+1)}
        P_bar = sum(pop_k[k] for k in pop_k)/len(pop_k)
        return float(max(abs(pop_k[k] - P_bar) for k in range(1,K+1)))/float(P_bar)

    def evaluate_n_majmin_districts(z_k, graph):
        min_minus_maj_list = [sum(graph.nodes[i]['p_min'] for i in z_k[k]) - sum(graph.nodes[i]['p_maj'] for i in z_k[k]) for k in z_k]

        min_minus_maj_districts = [k for k in z_k if sum(graph.nodes[i]['p_min'] for i in z_k[k]) > sum(graph.nodes[i]['p_maj'] for i in z_k[k])]
        fraction_minority_in_districts = {k:  sum(graph.nodes[i]['p_min'] for i in z_k[k])/sum(graph.nodes[i]['p_min']+graph.nodes[i]['p_maj'] for i in z_k[k]) for k in z_k }
        largest_minority_minority_distrct_margin = max([i[1] for i in fraction_minority_in_districts.items() if i[1] < majmin_threshold])
        racial_mix = {k: {} for k in z_k}
        fraction_hispanicmaj_in_districts = {k: sum(graph.nodes[i]['p_hispanic'] for i in z_k[k])/sum(graph.nodes[i]['pop'] for i in z_k[k]) for k in z_k }
        largest_minority_hispanic_distrct_margin = max([i[1] for i in fraction_hispanicmaj_in_districts.items() if i[1] < majmin_threshold])
        hispanic_maj_list = []
        for k in z_k:
            racial_mix[k]["maj"] = sum(graph.nodes[i]['p_maj'] for i in z_k[k])
            racial_mix[k]["min"] = sum(graph.nodes[i]['p_min'] for i in z_k[k])
            racial_mix[k]["total"] = racial_mix[k]["maj"] + racial_mix[k]["min"]
            racial_mix[k]["maj"] = racial_mix[k]["maj"]/racial_mix[k]["total"]
            racial_mix[k]["min"] = racial_mix[k]["min"]/racial_mix[k]["total"]
            racial_mix[k]["hispanic"] = sum(graph.nodes[i]['p_hispanic'] for i in z_k[k])/racial_mix[k]["total"]
            racial_mix[k]["black"] = sum(graph.nodes[i]['p_black'] for i in z_k[k])/racial_mix[k]["total"]
            racial_mix[k]["amindian"] = sum(graph.nodes[i]['p_amindian'] for i in z_k[k])/racial_mix[k]["total"]
            racial_mix[k]["asian"] = sum(graph.nodes[i]['p_asian'] for i in z_k[k])/racial_mix[k]["total"]
            racial_mix[k]["pacific"] = sum(graph.nodes[i]['p_pacific'] for i in z_k[k])/racial_mix[k]["total"]
            if racial_mix[k]["hispanic"] > majmin_threshold:
                hispanic_maj_list.append(k)

        return len(hispanic_maj_list), min_minus_maj_districts, fraction_minority_in_districts, largest_minority_hispanic_distrct_margin, racial_mix


    def evaluate_strong_dists(z_k, graph, margin_of_victory):
        pop_rep_k = {k: sum(graph.nodes[i]['p_rep'] for i in z_k[k]) for k in z_k}
        pop_dem_k = {k: sum(graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
        pop_voted_k = {k: pop_rep_k[k]+pop_dem_k[k]  for k in z_k}
        return [k for k in z_k if float(pop_rep_k[k]-pop_dem_k[k])/pop_voted_k[k] >= margin_of_victory], [k for k in z_k if float(pop_dem_k[k]-pop_rep_k[k])/pop_voted_k[k] >= margin_of_victory]


    def evaluate_hierarchical_competitiveness(z_k,graph,cmpttv_threshold):
        pop_rep_k = {k: sum(graph.nodes[i]['p_rep'] for i in z_k[k]) for k in z_k}
        pop_dem_k = {k: sum(graph.nodes[i]['p_dem'] for i in z_k[k]) for k in z_k}
        margin_k = {k: float(abs(pop_rep_k[k]-pop_dem_k[k]))/(pop_rep_k[k]+pop_dem_k[k]) for k in z_k}
        num_cmpttv_dists = len([k for k in margin_k if margin_k[k] <= cmpttv_threshold])
        if len(z_k) == num_cmpttv_dists: #If all the districts are competitive
            max_margin_noncmpttv = 0
        else:
            max_margin_noncmpttv = max([margin_k[k] for k in z_k if margin_k[k] > cmpttv_threshold])

        return num_cmpttv_dists, max_margin_noncmpttv


    def evaluate_number_elections_for_each_party(z_k, graph):
        fields = ['G18USSDSIN', 'G18GOVDGAR', 'G18SOSDHOB', 'G18ATGDCON', 'G18TREDMAN', 'G18SPIDHOF', 'G18MNIDPIE', 'G20PREDBID','G20USSDKEL']
        fields += ['G18USSRMCS', 'G18GOVRDUC', 'G18SOSRGAY', 'G18ATGRBRN', 'G18TRERYEE', 'G18SPIRRIG', 'G18MNIRHAR', 'G20PRERTRU','G20USSRMCS']

        actual_elections = ['G18USS', 'G18GOV', 'G18SOS', 'G18ATG', 'G18TRE', 'G18SPI', 'G18MNI', 'G20PRE', 'G20USS']
        count_n_election_wins = {k: [0,0] for k in z_k}
        for k in z_k:
            for election in actual_elections:
                for field in fields:
                    if field[:6] == election:
                        if field[6] == 'R':
                            pop_rep = sum(graph.nodes[i][field] for i in z_k[k])
                        elif field[6] == 'D':
                            pop_dem = sum(graph.nodes[i][field] for i in z_k[k])
                if pop_rep < pop_dem:
                    count_n_election_wins[k][0] += 1
                elif pop_rep > pop_dem:
                    count_n_election_wins[k][1] += 1

        return count_n_election_wins


class algorithm:

    def vickrey_initial_solution_heuristic(graph):
        print("Stage 1: Generating an initial solution")
        pop = {i: graph.nodes[i]['pop'] for i in list(graph.nodes())}
        P_bar = sum(pop[i] for i in pop)/K
        iterations = 0
        while(1):
            z_k, z_i = {}, {}
            district_pop = {}
            unassigned_units = list(graph.nodes())
            """ Initial creation of districts """
            k = 1
            while(unassigned_units != []):
                if k not in z_k:
                    first_unit = random.choice(unassigned_units)
                    z_k[k] = [first_unit]
                    z_i[first_unit] = k
                    unassigned_units.remove(first_unit)
                    district_pop[k] = pop[first_unit]
                else:
                    candidate_units = list(set(unassigned_units)&set([j for i in z_k[k] for j in graph.neighbors(i) if pop[j]+district_pop[k] <= P_bar*(1+tau)]))
                    if candidate_units != []:
                        next_unit = random.choice(candidate_units)
                        z_k[k].append(next_unit)
                        z_i[next_unit] = k
                        unassigned_units.remove(next_unit)
                        district_pop[k] += pop[next_unit]
                    else:
                        k += 1

            least_pop =  min([val for key,val in list(district_pop.items())])
            most_pop =  max([val for key,val in list(district_pop.items())])

            """ Iterative merging till K districts """
            while(len(z_k) > K):
                least_pop_distr = min(district_pop, key=district_pop.get)
                neighboring_distrs = list(set([z_i[j] for i in z_k[least_pop_distr] for j in graph.neighbors(i)]))
                least_pop_neighbor_distr = min({key: val for key,val in list(district_pop.items()) if key in neighboring_distrs and key != least_pop_distr}, key=district_pop.get)
                merged_distr = min(least_pop_neighbor_distr,least_pop_distr)
                other_distr = max(least_pop_neighbor_distr,least_pop_distr)
                z_k[merged_distr] = z_k[least_pop_neighbor_distr] + z_k[least_pop_distr]
                for i in z_k[least_pop_distr]:
                    z_i[i] = merged_distr
                for i in z_k[least_pop_neighbor_distr]:
                    z_i[i] = merged_distr
                district_pop[merged_distr] = district_pop[least_pop_neighbor_distr] + district_pop[least_pop_distr]
                z_k.pop(other_distr, None)
                district_pop.pop(other_distr, None)
            z_k_copy, z_i_copy = {}, {}
            k = 1
            for k_dash in z_k:
                z_k_copy[k] = z_k[k_dash]
                for i in z_k[k_dash]:
                    z_i_copy[i] = k
                k += 1
            z_k = z_k_copy.copy()
            z_i = z_i_copy.copy()

            """ Local search improvement """
            z_k, z_i, pop_bal, movement_log = algorithm.local_search(z_k, z_i, graph, 'pop_bal', {})
            print("pop bal after local search:",pop_bal)

            district_pop = {k: sum(pop[i] for i in z_k[k]) for k in z_k}
            least_pop =  min([val for key,val in list(district_pop.items())])
            most_pop =  max([val for key,val in list(district_pop.items())])
            iterations += 1

            if (least_pop >= P_bar*(1-tau)) and (most_pop <= P_bar*(1+tau)):
                return z_i, z_k


    def local_search_recombination(z_k_initial, z_i_initial, graph, criteria, max_iteration, number_of_majority_minority_dists_needed, n_iterations_no_improvement=9999999, print_log=True, heuristic_run_time=36000, lb_ncompetitive_needed=0, ub_cmpptv_needed=0, ub_hier_cmpttv_needed=0,ub_compactness_needed=0):
        """ Initialization """
        P_bar = sum(graph.nodes[i]['pop'] for i in graph.nodes())/K
        z_i_best = z_i_initial.copy()
        z_k_best = z_k_initial.copy()
        if criteria == 'pop_bal':
            obj_best = metric.evaluate_population_balance(z_k_best,graph)
        elif criteria == 'mincut':
            obj_best = metric.evaluate_compactness_edgecuts(z_i_best,graph)
        elif criteria == 'maxmargin':
            obj_best = metric.evaluate_max_margin(z_k_best,graph)
        elif criteria == 'hier_cmpttv':
            num_cmpttv, max_margin_non_cmmptv = metric.evaluate_hierarchical_competitiveness(z_k_best,graph,cmpttv_threshold)
            obj_best = - num_cmpttv + max_margin_non_cmmptv
        elif criteria == 'VRA':
            obj_best = - metric.evaluate_n_majmin_districts(z_k_best,graph)[3]

        local_search_start_time = time.time()
        print("Starting improvement using Deford local search to optimize for",criteria)
        print("starting obj:",obj_best)
        obj_iterations, objbest_iterations = [], []
        iteration = 0
        last_improvement_iteration = 0
        while((((time.time()-local_search_start_time) <= heuristic_run_time) and iteration < max_iteration)):
            """ Identifying two random neighboring districts """
            k1 = random.choice(range(1,K+1))
            neighboring_units_of_k1 = reduce(lambda x, y: x+y, [list(graph.neighbors(i)) for i in z_k_best[k1]])
            neighboring_districts_of_k1 = list(set([z_i_best[i] for i in neighboring_units_of_k1 if z_i_best[i]!=k1]))

            districts_to_merge = random.sample(neighboring_districts_of_k1, 1)

            """ Merging the districts """
            merged_district = list(z_k_best[k1])
            for k in districts_to_merge:
                merged_district += list(z_k_best[k])

            total_merged_population = sum(graph.nodes[i]['pop'] for i in merged_district)
            remaining_subtree = merged_district
            z_i_new, z_k_new = {}, z_k_best.copy()
            feasibilities = []
            feasibilities_ctgs = []
            district_count = 1
            for k in districts_to_merge:
                P_minimum = max(P_bar*(1-tau), total_merged_population-P_bar*(1+tau))
                P_maximum = min(P_bar*(1+tau), total_merged_population-P_bar*(1-tau))
                # new_district = gerrychain.bipartition_tree(graph.subgraph(remaining_subtree), 'pop', P_bar, tau, 1)
                new_district = gerrychain.bipartition_tree_using_bounds(graph.subgraph(remaining_subtree), 'pop', P_minimum, P_maximum, 1)

                z_k_new[k] = new_district
                remaining_subtree = list(set(remaining_subtree) - set(new_district))

                district_pop = sum(graph.nodes[i]['pop'] for i in new_district)
                feasibilities.append((abs(district_pop - P_bar)/P_bar < tau))
                district_count += 1

            z_k_new[k1] = remaining_subtree
            district_pop = sum(graph.nodes[i]['pop'] for i in remaining_subtree)
            feasibilities.append((abs(district_pop - P_bar)/P_bar < tau))

            for k in z_k_new:
                for i in z_k_new[k]:
                    z_i_new[i] = k

            """ Check if new district plan satisfies criteria """
            pop_bal = metric.evaluate_population_balance(z_k_new,graph)
            # print("New population balance:",pop_bal)
            feasibility_criteria = {'sumdist': 0, 'mincut': 0, 'pop_bal': 0, 'VRA': 0, 'maxmargin': 0, 'hier_cmpttv':0}
            feasibility_criteria[criteria] = (pop_bal < tau)
            if criteria in ['sumdist', 'mincut', 'VRA', 'maxmargin','hier_cmpttv']:
                if ub_cmpptv_needed > 0:
                    max_margin = metric.evaluate_max_margin(z_k_new,graph)
                    feasibility_criteria['sumdist'] = feasibility_criteria['sumdist'] and max_margin <= ub_cmpptv_needed
                    feasibility_criteria['mincut'] = feasibility_criteria['mincut'] and max_margin <= ub_cmpptv_needed

                if number_of_majority_minority_dists_needed > 0:
                    feasibility_criteria['sumdist'] = feasibility_criteria['sumdist'] and metric.evaluate_n_majmin_districts(z_k_new,graph)[0] >= number_of_majority_minority_dists_needed
                    feasibility_criteria['mincut'] = feasibility_criteria['mincut'] and metric.evaluate_n_majmin_districts(z_k_new,graph)[0] >= number_of_majority_minority_dists_needed
                    feasibility_criteria['VRA'] = feasibility_criteria['VRA'] and metric.evaluate_n_majmin_districts(z_k_new,graph)[0] >= number_of_majority_minority_dists_needed
                    feasibility_criteria['maxmargin'] = feasibility_criteria['maxmargin'] and metric.evaluate_n_majmin_districts(z_k_new,graph)[0] >= number_of_majority_minority_dists_needed
                    feasibility_criteria['hier_cmpttv'] = feasibility_criteria['hier_cmpttv'] and metric.evaluate_n_majmin_districts(z_k_new,graph)[0] >= number_of_majority_minority_dists_needed

                if ub_hier_cmpttv_needed != 0:
                    num_cmpttv, max_margin_non_cmmptv = metric.evaluate_hierarchical_competitiveness(z_k_new,graph,cmpttv_threshold)
                    hier_cmpttv = - num_cmpttv + max_margin_non_cmmptv
                    feasibility_criteria['sumdist'] = feasibility_criteria['sumdist'] and hier_cmpttv >= ub_hier_cmpttv_needed
                    feasibility_criteria['mincut'] = feasibility_criteria['mincut'] and hier_cmpttv >= ub_hier_cmpttv_needed

                if ub_compactness_needed != 0:
                    compactness = metric.evaluate_compactness_edgecuts(z_i_new, graph)
                    feasibility_criteria['maxmargin'] = feasibility_criteria['maxmargin'] and compactness <= ub_compactness_needed
                    feasibility_criteria['hier_cmpttv'] = feasibility_criteria['hier_cmpttv'] and compactness <= ub_compactness_needed

            if feasibility_criteria[criteria]:
                if criteria == 'pop_bal':
                    obj = pop_bal
                elif criteria == 'effgap':
                    obj =  metric.evaluate_effgap(z_k_new, graph)
                elif criteria == 'mincut':
                    obj = metric.evaluate_compactness_edgecuts(z_i_new, graph)
                    # print("Iteration",iteration,"Mincut obj after recombination", obj, "max margin:", metric.evaluate_max_margin(z_k_new,graph))
                elif criteria == 'maxmargin':
                    obj = metric.evaluate_max_margin(z_k_new,graph)
                elif criteria == 'hier_cmpttv':
                    num_cmpttv, max_margin_non_cmmptv = metric.evaluate_hierarchical_competitiveness(z_k_new,graph,cmpttv_threshold)
                    obj = - num_cmpttv + max_margin_non_cmmptv
                elif criteria == 'VRA':
                    obj = - metric.evaluate_n_majmin_districts(z_k_new,graph)[3]
                    # print("Iteration",iteration,"VRA obj after recombination:", obj, "No. of VRA districts:", metric.evaluate_n_majmin_districts(z_k_new,graph)[0])
                elif criteria== 'sumdist':
                    obj, district_centers_k_modified = metric.evaluate_compactness_sumdist(z_k_new, graph, d_matrix)

                obj_iterations.append(obj)
                if obj <= obj_best:
                    if obj < obj_best:
                        last_improvement_iteration = iteration
                    z_i_best = z_i_new.copy()
                    z_k_best = z_k_new.copy()
                    obj_best = obj
                    if print_log:
                        print("Iteration",iteration,"New best obj:", obj,"max margin:", metric.evaluate_max_margin(z_k_best,graph), "No. of VRA districts:", metric.evaluate_n_majmin_districts(z_k_best,graph)[0])

                objbest_iterations.append(obj_best)

                if criteria == 'VRA' and metric.evaluate_n_majmin_districts(z_k_new,graph)[0] > number_of_majority_minority_dists_needed:
                    z_i_best = z_i_new.copy()
                    z_k_best = z_k_new.copy()
                    obj_best = obj
                    print("Iteration",iteration,"Exiting with no. of VRA districts:", metric.evaluate_n_majmin_districts(z_k_best,graph)[0])
                    break

                elif criteria == 'hier_cmpttv' and obj < - K+1:
                    z_i_best = z_i_new.copy()
                    z_k_best = z_k_new.copy()
                    obj_best = obj
                    print("Iteration",iteration,"Exiting with no. of cmpttv districts:", -obj)
                    break

                if n_iterations_no_improvement > 0 and iteration - last_improvement_iteration > n_iterations_no_improvement:
                    break

            iteration += 1


        return z_k_best, z_i_best, obj_best, obj_iterations, objbest_iterations


    def local_search(z_k_initial, z_i_initial, graph, criteria, d_matrix, heuristic_run_time=360000, print_log=False, ub_hier_cmpttv_needed=0, ub_compactness_needed=0, n_iterations_no_improvement = 1000, vickrey_initial = True):

        def contiguity_check_after_move_neighbourhood(z_k, unit, from_district, to_district, graph):
            flag = 0
            # unit_neighborhood_fromdistrict = list(set(graph.neighbors(unit)).intersection(set(z_k[from_district])))
            # unit_neighborhood_fromdistrict_subgraph = graph.subgraph(unit_neighborhood_fromdistrict)
            # # unit_neighborhood_fromdistrict_subgraph = graph.subgraph(z_k[from_district])
            # if len(unit_neighborhood_fromdistrict_subgraph) != 0:
            #     flag = nx.is_connected(unit_neighborhood_fromdistrict_subgraph)
            #     # print(flag, nx.is_connected(graph.subgraph(set(z_k[from_district])-{unit})))
            # else:
            #     flag = 0

            subgraph = graph.subgraph(set(z_k[from_district])-{unit})
            if len(subgraph) != 0:
                flag = nx.is_connected(subgraph)
            else:
                flag = 0
            # print(len(graph.subgraph(set(z_k[from_district])-{unit}).nodes()))
            # print "Moving unit",unit,"from district",from_district,"to district",to_district,"and flag_1 is",flag_1

            return flag

        def population_balance_check(pop_k, unit, from_district, to_district, graph):
            flag = 1
            from_district_pop = pop_k[from_district] - graph.nodes[unit]['pop']
            to_district_pop = pop_k[to_district] + graph.nodes[unit]['pop']
            if from_district_pop >= pop_bal_lb and from_district_pop <= pop_bal_ub and to_district_pop >= pop_bal_lb and to_district_pop <= pop_bal_ub:
                flag = 1
            else:
                flag = 0
            return flag

        def evaluate_population_balance(pop_k,graph):
            # return float(max(abs(pop_k[k] - P_bar) for k in range(1,K+1)))/float(P_bar)
            return float(max(abs(pop_k[k] - P_bar) for k in range(1,K+1)))

        def evaluate_incremental_population_balance(pop_k, unit, from_district, to_district, graph):
            new_pop_k = pop_k.copy()
            new_pop_k[from_district] = new_pop_k[from_district] -  graph.nodes[unit]['pop']
            new_pop_k[to_district] = new_pop_k[to_district] +  graph.nodes[unit]['pop']
            # return float(max(abs(new_pop_k[k] - P_bar) for k in range(1,K+1)))/float(P_bar)
            return float(max(abs(new_pop_k[k] - P_bar) for k in range(1,K+1)))

        def evaluate_incremental_compactness_edgecuts(z_k,current_compactness, unit, from_district, to_district, graph):
            final_compactness = current_compactness
            for j in set(graph.neighbors(unit)).intersection(set(z_k[from_district])):
                final_compactness += graph[unit][j]['bndry']
                # final_compactness += 1
            for j in set(graph.neighbors(unit)).intersection(set(z_k[to_district])):
                final_compactness -= graph[unit][j]['bndry']
                # final_compactness -= 1

            # final_compactness = final_compactness + len(set(graph.neighbors(unit)).intersection(set(z_k[from_district])))
            # final_compactness = final_compactness - len(set(graph.neighbors(unit)).intersection(set(z_k[to_district])))
            return final_compactness


        z_i_best = z_i_initial.copy()
        z_k_best = z_k_initial.copy()
        pop_k = {k: sum(graph.nodes[i]['pop'] for i in z_k_best[k]) for k in range(1,K+1)}

        if criteria == 'pop_bal':
            obj_best = evaluate_population_balance(pop_k,graph)
            current_compactness = metric.evaluate_compactness_edgecuts(z_i_best,graph)
        candidate_moves = []
        iteration, iteration_last_improvement = 0, 0
        local_search_start_time = time.time()
        movement_log = []
        if print_log:
            print("Starting improvement from", obj_best)

        candidate_moves = []
        while((candidate_moves!=[] or iteration==0) and ((time.time()-local_search_start_time) <= heuristic_run_time)):
            candidate_moves = []
            border_edges = [(i,j) for (i,j) in graph.edges() if z_i_best[j] != z_i_best[i]]
            i, k = random.choice(border_edges)

            z_i = z_i_best.copy()
            z_i[i] = k

            for i in list(graph.nodes()):
                for k in set([z_i_best[j] for j in list(graph.neighbors(i)) if z_i_best[j]!= z_i_best[i]]):
                    pop_bal = evaluate_incremental_population_balance(pop_k, i, z_i_best[i], k, graph)
                    feasibility_criteria = {criteria: contiguity_check_after_move_neighbourhood(z_k_best, i, z_i_best[i], k, graph)}

                    if criteria == 'pop_bal' and vickrey_initial==False: # if we are optimizing population balance in stage 4

                        z_i_aftermove = {i_prime: z_i_best[i_prime] for i_prime in z_i_best}
                        z_i_aftermove[i] = k

                        z_k_aftermove = {k_prime: [i_prime for i_prime in z_k_best[k_prime]] for k_prime in z_k_best}
                        z_k_aftermove[z_i_best[i]].remove(i)
                        z_k_aftermove[k].append(i)

                        if ub_hier_cmpttv_needed != 0:
                            num_cmpttv, max_margin_non_cmmptv = metric.evaluate_hierarchical_competitiveness(z_k_aftermove,graph,cmpttv_threshold)
                            hier_cmpttv = - num_cmpttv + max_margin_non_cmmptv
                            feasibility_criteria[criteria] = feasibility_criteria['pop_bal'] and hier_cmpttv <= ub_hier_cmpttv_needed

                        if ub_compactness_needed > 0:
                            compactness_incremental = evaluate_incremental_compactness_edgecuts(z_k_best, current_compactness, i, z_i_best[i], k, graph)
                            feasibility_criteria[criteria] = feasibility_criteria[criteria] and compactness_incremental <= ub_compactness_needed

                    if feasibility_criteria[criteria]:
                        if criteria == 'pop_bal':
                            obj = pop_bal
                            # print("Obj after moving unit",i,":", obj)
                        if (vickrey_initial==True and obj < obj_best) or (vickrey_initial==False and obj <= obj_best):
                            if obj < obj_best:
                                iteration_last_improvement = iteration
                            candidate_moves.append(i)
                            i_best = i
                            k_best = k
                            from_dist = z_i_best[i_best]
                            z_k_best[k_best].append(i_best)
                            z_k_best[z_i_best[i_best]].remove(i_best)
                            pop_k[z_i_best[i_best]] = pop_k[z_i_best[i_best]] - graph.nodes[i_best]['pop']
                            pop_k[k_best] = pop_k[k_best] + graph.nodes[i_best]['pop']
                            z_i_best[i_best] = k_best
                            obj_best = obj
                            if criteria == 'pop_bal' and print_log:
                                print("iteration", iteration,": unit", i_best, "is moved from district", from_dist,"to district", k_best,"with best population balance obj", obj_best)
                                movement_log.append([iteration,i_best,from_dist,k_best,obj_best])
                            if ub_compactness_needed > 0:
                                current_compactness = compactness_incremental

                        iteration += 1

                        if (iteration-iteration_last_improvement >= n_iterations_no_improvement) or (criteria == 'pop_bal' and obj_best<=1.01):
                            print("Terminating after",iteration,"iterations")
                            return z_k_best, z_i_best, obj_best, movement_log

        return z_k_best, z_i_best, obj_best, movement_log



    def uncoarsen_with_local_search(z_sol, z_k, level_l_to_l_plus_1, level_l_plus_1_to_l, county_graph, tract_graph, level_graph_l, max_iter, number_of_majority_minority_dists_needed, n_iterations_no_improvement=999999, criteria = 'mincut', perform_local_search = True, print_log=True, streamlit_print=False):
        if level_l_plus_1_to_l == {}:
            return z_sol,z_k, 0, 0
        n_levels = max(level_l_plus_1_to_l.keys())
        """ Uncoarsening in the decreasing order of levels from n_levels through 0 """
        z_i_level_l = {n_levels: z_sol.copy()}
        z_k_level_l = {n_levels: z_k.copy()}
        obj_best_final = {}
        objbest_iterations_alllevels = {}
        for l in sorted(list(range(0,n_levels)),reverse = 1):
            z_i_level_l[l] = {}
            z_k_level_l[l] = {}

            for i in z_i_level_l[l+1]:
                for i_lower in level_l_plus_1_to_l[l+1][i]:
                    z_i_level_l[l][i_lower] = z_i_level_l[l+1][i]

            for k in range(1,K+1):
                z_k_level_l[l][k] = []

                for i in z_k_level_l[l+1][k]:                       # for every upper (l+1) level unit i assigned to district k
                    for i_lower in level_l_plus_1_to_l[l+1][i]:     # for every lower (l) level unit i_lower uncoarsened from i
                        z_k_level_l[l][k].append(i_lower)           # assign i_lower to district k in lower level l

            """ Local search improvement at level l """
            objbest_iterations_alllevels[l], obj_best_final[l] = [], []
            if perform_local_search:
                z_k_improved, z_i_improved, obj_best_final[l], obj_iterations, objbest_iterations_alllevels[l] = algorithm.local_search_recombination(z_k_level_l[l], z_i_level_l[l], level_graph_l[l], criteria, max_iter, number_of_majority_minority_dists_needed, n_iterations_no_improvement, print_log) #'mincut', 'sumdist'
                z_k_level_l[l], z_i_level_l[l] = z_k_improved.copy(), z_i_improved.copy()
            print("Compactness objective at level %i of uncoarsening:"%l,obj_best_final[l])
            if streamlit_print:
                st.write("Compactness at level",l," of uncoarsening:",int(obj_best_final[l]),"miles after",len(obj_iterations),"local search steps.")

        # outlet.plot_map(level_graph_l[0], z_k_level_l[0], z_i_level_l[0], range(1,K+1), [], [])
        return z_i_level_l[0], z_k_level_l[0], objbest_iterations_alllevels, obj_best_final


class outlet:
    def write_map_to_file(filename, z_i):
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['GEOID', 'district'])
            for key, value in list(z_i.items()):
                writer.writerow([key, value])


    def print_metrics(z_k_tract, z_i_tract, graph, P_bar):
        print("Metrics:")
        print("Overall R vote-share:", float(sum(graph.nodes[i]['p_rep'] for i in graph.nodes()))/(sum(graph.nodes[i]['p_rep']+graph.nodes[i]['p_dem'] for i in graph.nodes())))

        pop_k = {k: sum(graph.nodes[i]['pop'] for i in z_k_tract[k]) for k in z_k_tract}
        print("Population balance:", metric.evaluate_population_balance(z_k_tract, graph), [pop_k[k] for k in pop_k])
        compactness = metric.evaluate_compactness_edgecuts(z_i_tract, graph)
        print("Compactness:", compactness)
        max_margin = metric.evaluate_max_margin(z_k_tract, graph)
        print("Max margin (cmpttv):", max_margin)
        print("No. of cmpttv districts:", len(metric.evaluate_number_of_cmptitv_dists(z_k_tract, graph, 0.07)[0]))
        print("Cmpttv districts:", metric.evaluate_number_of_cmptitv_dists(z_k_tract, graph, 0.07)[0])
        print("Strong D districts:", metric.evaluate_strong_dists(z_k_tract, graph, 0.07)[1])
        print("Simple D districts:", metric.evaluate_strong_dists(z_k_tract, graph, 0)[1])
        print("Strong R districts:", metric.evaluate_strong_dists(z_k_tract, graph, 0.07)[0])
        print("Simple R districts:", metric.evaluate_strong_dists(z_k_tract, graph, 0)[0])
        print("Majority minority districts:", metric.evaluate_n_majmin_districts(z_k_tract, graph)[1])
        racial_mix = metric.evaluate_n_majmin_districts(z_k_tract, graph)[4]
        print("Minority fraction in maj_min districts:", metric.evaluate_n_majmin_districts(z_k_tract, graph)[2])
        n_wins = metric.evaluate_number_elections_for_each_party(z_k_tract, graph)
        margin, who_won = {}, {}
        for k in z_k_tract:
            dem_vote_share = sum(graph.nodes[i]['p_dem'] for i in z_k_tract[k])/sum(graph.nodes[i]['p_dem']+graph.nodes[i]['p_rep'] for i in z_k_tract[k])
            margin[k] = abs(0.5-dem_vote_share)*2
            if dem_vote_share > 0.5:
                who_won[k] = 'D'
            else:
                who_won[k] = 'R'
        solution = pd.DataFrame()
        solution['District'] = range(1,K+1)
        solution['Population'] = [pop_k[k] for k in range(1,K+1)]
        solution['Population deviation'] = [int(abs(P_bar-pop_k[k])) for k in range(1,K+1)]
        solution['Hispanic'] = [round(100*racial_mix[k]['hispanic'],2) for k in range(1,K+1)]
        solution['NH White'] = [round(100*racial_mix[k]['maj'],2) for k in range(1,K+1)]
        solution['Other minority'] = [round(100*racial_mix[k]['min']-100*racial_mix[k]['hispanic'],2) for k in range(1,K+1)]
        solution['Vote Spread'] = [round(100*margin[k],2) for k in range(1,K+1)]
        solution['Dem. Wins'] = [n_wins[k][0] for k in range(1,K+1)]
        solution['Rep. Wins'] = [n_wins[k][1] for k in range(1,K+1)]
        solution['Party lean'] = [who_won[k] for k in range(1,K+1)]

        opt_effgap = metric.evaluate_effgap(z_k_tract,graph)
        passymetry_opt = metric.compute_partisan_symmetry(z_k_tract, graph, 0)
        max_margin = metric.evaluate_max_margin(z_k_tract, graph)
        majmin =  metric.evaluate_n_majmin_districts(z_k_tract, graph)[0]
        compactness = metric.evaluate_compactness_edgecuts(z_i_tract, graph)
        n_cmpttv = len(metric.evaluate_number_of_cmptitv_dists(z_k_tract, graph, 0.07)[0])
        return opt_effgap, passymetry_opt, max_margin, majmin, compactness, n_cmpttv, solution


    def plot_map(graph, z_k, z_i, districts_to_represent, vertices_to_label, matched_edges):
        pos = {i:(graph.nodes[i]['x'], graph.nodes[i]['y']) for i in graph.nodes()}
        pos_labels = {i:(graph.nodes[i]['x'], graph.nodes[i]['y']-.02) for i in vertices_to_label}
        color_k = {1:'b', 2:'g', 3:'r', 4: 'c', 5:'m', 6:'y', 7:'k', 8: 'w', 9:'red'}
        for k in districts_to_represent:
            nx.draw_networkx_nodes(graph,pos, nodelist=z_k[k], node_color=color_k[k], label = {i: i for i in vertices_to_label}, node_size = 20)
            for (i,j) in graph.edges():
                if z_i[i] == k and z_i[j] == k:
                    if ((i,j) in matched_edges) or ((j,i) in matched_edges):
                        nx.draw_networkx_edges(graph,pos, edgelist = [(i,j)], edge_color = color_k[k], style = 'dashed')
                    else:
                        nx.draw_networkx_edges(graph,pos, edgelist = [(i,j)], edge_color = color_k[k])


        for (i,j) in graph.edges():
            if z_i[i] != z_i[j]:
                if ((i,j) in matched_edges) or ((j,i) in matched_edges):
                    nx.draw_networkx_edges(graph,pos, edgelist = [(i,j)], edge_color = 'r', style = 'dashed')
                else:
                    nx.draw_networkx_edges(graph,pos, edgelist = [(i,j)], edge_color = 'k', style = 'dotted')

        nx.draw_networkx_labels(graph.subgraph(vertices_to_label),pos = pos_labels, labels = {i: str(i)[-5:] for i in vertices_to_label}, font_size = 10)
        plt.axis('off')
        plt.show() # display





""" Inputs """
K = 9     # number of districts
n_counties_to_split = 7    # number of counties to split
tau = .01     # population deviation threshold for stages 1 through 3
cmpttv_threshold = .07    # vote-margin within which a district is considered competitive
number_of_majority_minority_dists_needed = 2    # number of majority-minority districts
majmin_threshold = 0.5   # minimum fractional minority population for a district to be considered majority-minority


""" Algorithm parameters """
n_levels = 1   # Number of coarsening levels in the multilevel algorithm
max_iterations = 1000000    # A large number upper bound on the permissible local search iterations
n_iterations_no_improvement = 500  # Maximum number of consecutive non-improving local search iterations
compromise_factor = 0.1    # Compromise factor for compactness

n_maps_to_draw = 100  # number of maps to generate for stages 1-3


def main():
    global county_graph, tract_graph, P_bar, communities_dict

    """ Read the data """
    county_graph = input.read_data_county_2020() # read county-level data
    tract_graph = input.read_data_tract_2020() # read census tract-level data
    P_bar = int(sum(tract_graph.nodes[i]['pop'] for i in tract_graph.nodes())/K) # average district population
    communities_dict, places_dict = input.read_communities_of_interest_and_places() # read data on communities of interest and places to preserve

    """ Uncomment the following for demographic info """

    print("Number of census tracts:",len(tract_graph.nodes()),", number of edges:", len(tract_graph.edges()))
    print("Fraction of Democrats:",sum(tract_graph.nodes[i]['p_dem'] for i in tract_graph.nodes())/sum(tract_graph.nodes[i]['p_dem']+tract_graph.nodes[i]['p_rep'] for i in tract_graph.nodes()))
    print("Fraction of Hispanic populace:",sum(tract_graph.nodes[i]['p_hispanic'] for i in tract_graph.nodes())/sum(tract_graph.nodes[i]['pop'] for i in tract_graph.nodes()))
    print("Number of communities of interest:",len(communities_dict),"with %i census tracts"%len(set([item for sublist in list(communities_dict.values()) for item in sublist])))
    print("Number of places:",len(places_dict),"with %i census tracts"%len(set([item for sublist in list(places_dict.values()) for item in sublist])))
    print("Average district population:",P_bar,"lower bound:",(1-tau)*P_bar,"upper bound:",(1+tau)*P_bar)


    """ Pre-processing: create Hybrid Census Tract graph """

    hybrid_tract_graph, label_community_tract, label_tract_community = multilevel_algo.create_hybrid_graph(county_graph, tract_graph, n_counties_to_split, communities_dict, places_dict,'tract')

    print("Hybrid tract graph has %i units and %i edges"%(len(hybrid_tract_graph.nodes()),len(hybrid_tract_graph.edges())))


    """ Stage 0: Coarsening """
    coarse_graph, coarse_graph_before_aggregation, level_l_to_l_plus_1, level_l_plus_1_to_l, level_graph_l = multilevel_algo.aggregate_multilevel(tract_graph, hybrid_tract_graph, label_community_tract, n_levels, plot_coarse_levels=False)


    """ Draw 100 maps """
    for map_id in range(1,n_maps_to_draw+1):
        print("Starting to draw map",map_id)

        """ Stage 1: Initial feasible district map"""


        """ Stage 1 Phase 1: Find a contiguous and 1%-balanced map """

        start_time_initial_heuristic = time.time()
        z_i, z_k = algorithm.vickrey_initial_solution_heuristic(coarse_graph)
        time_vickrey = time.time() - start_time_initial_heuristic

        """ Stage 1 Phase 2: Find VRA districts at coarse level """

        start_time_VRA = time.time()
        n_majmindists_current = metric.evaluate_n_majmin_districts(z_k, coarse_graph)[0]
        z_k_current, z_i_current = z_k.copy(), z_i.copy()
        print("Current number of VRA districts:",n_majmindists_current)
        n_restarts = 1
        while(n_majmindists_current < number_of_majority_minority_dists_needed):
            z_k, z_i, obj_outer, obj_iterations, objbest_iterations = algorithm.local_search_recombination(z_k_current, z_i_current, coarse_graph, 'VRA', max_iterations, n_majmindists_current, n_iterations_no_improvement, print_log=True)
            n_majmindists_new = metric.evaluate_n_majmin_districts(z_k, coarse_graph)[0]
            if n_majmindists_new > n_majmindists_current:
                n_majmindists_current = n_majmindists_new
                z_k_current, z_i_current = z_k.copy(), z_i.copy()
                print("Current number of VRA districts:",n_majmindists_current)
            else:
                n_restarts += 1
                if n_restarts%5==0:
                    z_i_current, z_k_current = algorithm.vickrey_initial_solution_heuristic(coarse_graph)
                    n_majmindists_current = metric.evaluate_n_majmin_districts(z_k_current, coarse_graph)[0]
                print("Restart #",n_restarts)

        time_VRA = time.time() - start_time_VRA


        """ Stage 2: Optimize for compactness """
        n_iterations_uncoarsening = max_iterations

        start_time_compactness = time.time()
        z_k, z_i, obj_outer, obj_iterations, objbest_iterations = algorithm.local_search_recombination(z_k_current, z_i_current, coarse_graph, 'mincut',  max_iterations, number_of_majority_minority_dists_needed, n_iterations_no_improvement)#, print_log=False)
        time_compactness = time.time() - start_time_compactness

        """  Uncoarsen the map to hybrid census tract level, optimize compactness in each level """
        start_time_uncoarsening = time.time()
        if n_levels > 0:
            z_i, z_k, objbest_iterations_alllevels, objbest_iterations = algorithm.uncoarsen_with_local_search(z_i, z_k, level_l_to_l_plus_1, level_l_plus_1_to_l, county_graph, tract_graph,level_graph_l, n_iterations_uncoarsening, number_of_majority_minority_dists_needed, n_iterations_no_improvement, criteria= 'mincut', print_log=True)
        time_uncoarsening = time.time() - start_time_uncoarsening


        """ Stage 3: Optimize for competitiveness """
        ub_compactness_needed = objbest_iterations[0]*(1+compromise_factor)
        print("Optimizing for competitiveness now. \n Relaxing compactness to be no more than",ub_compactness_needed)
        start_time_cmpttveness = time.time()
        z_k, z_i, obj_outer, obj_iterations, objbest_iterations = algorithm.local_search_recombination(z_k, z_i, hybrid_tract_graph, 'hier_cmpttv',  max_iterations, number_of_majority_minority_dists_needed, n_iterations_no_improvement, print_log=True, ub_hier_cmpttv_needed=0, ub_compactness_needed=ub_compactness_needed)
        time_cmpttveness = time.time() - start_time_cmpttveness

        pop_bal = metric.evaluate_population_balance(z_k, hybrid_tract_graph)

        opt_effgap, passymetry_opt, max_margin, majmin, compactness, n_cmpttv, df_solution = outlet.print_metrics(z_k, z_i, hybrid_tract_graph,P_bar)

        total_time=sum([time_vickrey,time_VRA,time_compactness,time_uncoarsening,time_cmpttveness])
        result = [map_id, tau, n_levels,  number_of_majority_minority_dists_needed, max_iterations,n_iterations_no_improvement,compromise_factor,time_vickrey,time_VRA,time_compactness,time_uncoarsening,time_cmpttveness,total_time,n_restarts]
        result += [opt_effgap, passymetry_opt, max_margin, n_cmpttv, majmin, compactness, pop_bal*P_bar]

        def csv_delimiter_fixer(filename):
            with open(filename, 'a+b') as fileobj:
                fileobj.seek(-1, 2)
                if fileobj.read(1) != b"\n":
                    fileobj.write(b"\r\n")
        csv_delimiter_fixer("results/aggregate_log.csv")


if __name__ == "__main__":
    main()
