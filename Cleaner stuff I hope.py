#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.graph_objects as go
import copy as cp
from pprint import pprint
from gurobipy import *
from random import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import time
from itertools import groupby,chain
import itertools
import pandas as pd
from pandas import *
import networkx as nx
import random
from random import shuffle
from itertools import chain
import string
import json
from dataclasses import dataclass
import uuid
import bisect
from datetime import datetime


# In[2]:


delta = 0.05
update_interval = 2
rcl_parameter = 0.3
llambda = 0.7


# In[51]:


graph_top_left = [x for x in chain(range(0,11), range(27,38),range(54,65),range(81,92),range(108,119),range(135,146),range(162,173),range(189,200),range(216,227),range(243,254),range(270,281),range(297,308),range(324,335),range(351,362))]
graph_top_right = [x for x in chain(range(378,389),range(405,416),range(432,443),range(459,470),range(486,497),range(513,524),range(540,551),range(567,578),range(594,605),range(621,632),range(648,658),range(675,686),range(702,712))]
graph_center = [x for x in chain(range(200,208), range(227,234),range(253,261),range(281,289),range(308,316),range(335,343),range(362,370),range(389,397),range(416,424),range(443,451),range(470,478),range(497,505),range(524,532))]
graph_left  = [x for x in chain(range(11,19), range(38,46),range(65,73),range(92,100),range(119,127),range(146,154),range(173,181))]
graph_right = [x for x in chain(range(551,559), range(578,586),range(605,613),range(632,640),range(659,667),range(686,694),range(713,721))]
graph_bottom_left = [x for x in chain(range(19,27), range(46,54),range(73,81),range(100,108),range(127,135),range(154,162),range(181,189),range(208,216),range(235,243),range(262,270),range(289,297),range(316,324),range(343,351),range(370,378))]
graph_bottom_right = [x for x in chain(range(397,405),range(424,432),range(451,459),range(478,486),range(505,513),range(532,540),range(559,567),range(586,594),range(613,621),range(640,648),range(667,675),range(694,702),range(721,728))]


# In[52]:


N = 27
G1 = nx.grid_2d_graph(N,N)

labels=dict(((i,j),i + (N-1-j)*N) for i, j in G1.nodes())
nx.relabel_nodes(G1,labels,False) #False=relabel the nodes in place
inds=labels.keys()
vals=labels.values()
inds=[(N-j-1,N-i-1) for i,j in inds]

#Create the dictionary of positions for the grid
grid_pos=dict(zip(vals,inds)) #Format: {node ID:(i,j)}

random.seed(511566511)
nodes = list(G1.nodes)
edges = list(G1.edges)

# top_left_remove = random.sample(graph_top_left,20)
# top_right_remove = random.sample(graph_top_right,10)
# bottom_left_remove = random.sample(graph_bottom_left,25)
# bottom_right_remove = random.sample(graph_bottom_right,35)
center_remove = random.sample(graph_center,50)
left_remove =random.sample(graph_left,50)
right_remove = random.sample(graph_right,50)

while nx.is_connected(G1) == True:
    while len(G1) > 500:
    

    #     G12.remove_nodes_from(top_left_remove)
    #     G12.remove_nodes_from(top_right_remove)
    #     G12.remove_nodes_from(bottom_left_remove)
    #     G12.remove_nodes_from(bottom_right_remove)
        G1.remove_nodes_from(center_remove)
        G1.remove_nodes_from(left_remove)
        G1.remove_nodes_from(right_remove)
        G1.remove_node(random.choice(list(G1.nodes)))
        if len(list(nx.isolates(G1)))>0:
            G1.remove_nodes_from(list(nx.isolates(G1)))
#         if nx.is_connected(G26) == False:
#             G26 = G26_C
#         else:
#             G26_C=G26.copy()
#     if len(G26) <= 500:
#             break


#Create the dictionary of positions for the grid
grid_pos=dict(zip(vals,inds)) #Format: {node ID:(i,j)}

G1.add_edge(201,203)
G1.add_edge(258,256)
G1.add_edge(231,233)
G1.add_edge(496,500)
G1.add_edge(523,527)
G1.add_edge(527,529)
G1.add_edge(422,424)
G1.add_edge(449,451)
G1.add_edge(391,389)
G1.add_edge(365,363)


#Clean the dictionaries in accordance with how we changed the original graph
for i in list(grid_pos):
    if i not in G1.nodes:
        grid_pos.pop(i)
        
coords = []
for i in inds:
    for key, value in grid_pos.items():
        if i == value:
            coords.append(i)


# In[ ]:





# In[4]:


N = 6
G1 = nx.grid_2d_graph(N,N)

labels=dict(((i,j),i + (N-1-j)*N) for i, j in G1.nodes())
nx.relabel_nodes(G1,labels,False) #False=relabel the nodes in place
inds=labels.keys()
vals=labels.values()
inds=[(N-j-1,N-i-1) for i,j in inds]

#Create the dictionary of positions for the grid
grid_pos=dict(zip(vals,inds)) #Format: {node ID:(i,j)}

random.seed(511566511)
nodes = list(G1.nodes)
edges = list(G1.edges)

# top_left_remove = random.sample(graph_top_left,20)
# top_right_remove = random.sample(graph_top_right,10)
# bottom_left_remove = random.sample(graph_bottom_left,25)
# bottom_right_remove = random.sample(graph_bottom_right,35)
# center_remove = random.sample(graph_center,50)
# left_remove =random.sample(graph_left,50)
# right_remove = random.sample(graph_right,50)

# while nx.is_connected(G1) == True:
#     while len(G1) > 20:
    


#         if len(list(nx.isolates(G1)))>0:
#             G1.remove_nodes_from(list(nx.isolates(G1)))
#         if nx.is_connected(G1) == False:
#             G1 = G1_C
#         else:
#             G1_C=G1.copy()
#         if len(G1) <= 20:
#                 break


#Create the dictionary of positions for the grid
grid_pos=dict(zip(vals,inds)) #Format: {node ID:(i,j)}


# G1.remove_node(11)
# G1.remove_node(8)
# G1.remove_node(18)


#Clean the dictionaries in accordance with how we changed the original graph
for i in list(grid_pos):
    if i not in G1.nodes:
        grid_pos.pop(i)
        
coords = []
for i in inds:
    for key, value in grid_pos.items():
        if i == value:
            coords.append(i)


# In[53]:


def get_furthest_nodes(G):
    sp_length = {} # dict containing shortest path distances for each pair of nodes
    diameter = None # will contain the graphs diameter (length of longest shortest path)
    furthest_node_list = [] # will contain list of tuple of nodes with shortest path equal to diameter
    
    for node in G.nodes:
        # Get the shortest path from node to all other nodes
        sp_length[node] = nx.single_source_dijkstra_path_length(G,node, weight = 'distance')
        longest_path = max(sp_length[node].values()) # get length of furthest node from node
        
        # Update diameter when necessary (on first iteration and when we find a longer one)
        if diameter == None:
            diameter = longest_path # set the first diameter
            
        # update the list of tuples of furthest nodes if we have a best diameter
        if longest_path >= diameter:
            diameter = longest_path
            
            # a list of tuples containing
            # the current node and the nodes furthest from it
            node_longest_paths = [(node,other_node)
                                      for other_node in sp_length[node].keys()
                                      if sp_length[node][other_node] == longest_path]
            if longest_path > diameter:
                # This is better than the previous diameter
                # so replace the list of tuples of diameter nodes with this nodes
                # tuple of furthest nodes
                furthest_node_list = node_longest_paths
            else: # this is equal to the current diameter
                # add this nodes tuple of furthest nodes to the current list    
                furthest_node_list = furthest_node_list + node_longest_paths
                
    # return the diameter,
        # all pairs of nodes with shortest path length equal to the diameter
        # the dict of all-node shortest paths
    return({'diameter':diameter,
            'furthest_node_list':furthest_node_list,
            'node_shortest_path_dicts':sp_length})

#Define a function to find the union of two lists
def Union(lst1,lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


# In[54]:


def plus_plus(ds, k):
    """
    Create cluster centroids using the k-means++ algorithm.
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """
#     np.random.seed(random_state)
    centroids = [random.choice(ds)]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        centroids.append(ds[i])

    return np.array(centroids)


# In[55]:


def construction_grasp(delta, rcl_parameter,llambda,graph_input):
    #Choosing centers

    locations = np.array(coords)

    centroids = plus_plus(locations, 10)

    centroids = centroids.tolist()

    centroids_tuple = []
    for i in centroids:
        centroids_tuple.append(tuple((i)))

    centers_depots = []
    for i in centroids_tuple:
        for key, value in grid_pos.items():
            if i == value:
                centers_depots.append(key)
                
    #Initialize randomized activities
    combinations = list(itertools.combinations(centers_depots, 2))



    #Calculate the average for each activity
    adjacent = {}
    for i in graph_input.nodes():
            adjacent[i] = []
    for e in graph_input.edges():
        adjacent[e[0]].append(e)
        adjacent[e[1]].append(e)

    #Define adjacent nodes for each node

    adjacent_nodes = {}
    nodes_new = {}
    for i in adjacent:
        adjacent_nodes[i] = []
        for e in range(len(adjacent[i])):
            adjacent_nodes[i].append(adjacent[i][e][0])
            adjacent_nodes[i].append(adjacent[i][e][1])
    for i in adjacent_nodes:
        nodes_new[i] = list(set(adjacent_nodes[i]))
    adjacent_nodes = {k:[vi for vi in v if k != vi] for k,v in nodes_new.items()}



    random.seed(2021)
    demand= {}
    for v in graph_input.nodes:
        demand[v] = random.randint(15,369)

    random.seed(2021)
    workload= {}
    for v in graph_input.nodes:
        workload[v] = random.randint(15,89)

    random.seed(2021)
    n_customers= {}
    for v in graph_input.nodes:
        n_customers[v] = random.randint(4,19)

    random.seed(2021)
    for v in centers_depots:
        demand[v] = 400
        workload[v] = 100
        n_customers[v] = 20
        for i in adjacent_nodes[v]:
            demand[i] = random.randint(370,400)
            workload[i] = random.randint(90,100)
            n_customers[i] = random.randint(15,20)

    random.seed(2021)
    distance= {}
    for e in graph_input.edges:
        distance[e] = random.randint(6,40)


    nx.set_node_attributes(graph_input, values = n_customers, name = "n_customers")
    nx.set_node_attributes(graph_input, values = demand, name = "demand")
    nx.set_node_attributes(graph_input, values = workload, name = "workload")
    nx.set_edge_attributes(graph_input, values = distance, name = "distance")
    
    
    shortest_paths_dict = get_furthest_nodes(graph_input)['node_shortest_path_dicts']
    graph_diameter = get_furthest_nodes(graph_input)['diameter']

    total_workload = 0 
    for v in graph_input.nodes:
        total_workload = total_workload + graph_input.nodes[v]['workload']
    average_workload = total_workload/len(centers_depots)

    total_customers = 0 
    for v in graph_input.nodes:
        total_customers = total_customers + graph_input.nodes[v]['n_customers']
    average_customers = total_customers/len(centers_depots)

    total_demand = 0 
    for v in graph_input.nodes:
        total_demand = total_demand + graph_input.nodes[v]['demand']
    average_demand = total_demand/len(centers_depots)



    selected_nodes = {}
    near_nodes = {}
    for k in centers_depots:
        selected_nodes[k] = []
        selected_nodes[k] = nx.ego_graph(graph_input,k, radius = 30, center=False, undirected=True, distance='distance')
        near_nodes[k] = list(selected_nodes[k].nodes())

#     for v in near_nodes:
#         print(list(any(v in val for val in near_nodes.values())))

    #Find the percentage of selected nodes from the graph
    num_nodes = 0
    for i in near_nodes:
        num_nodes = num_nodes+len(near_nodes[i])

    percentage_nodes = 1-(num_nodes/len(nodes))

    construction_time = time.time()

    #Create the initial districts by assigning the nodes in the neighborhood to depots
    district_customers = {}
    district_workload = {}
    district_demand = {}
    unassigned = graph_input.nodes
    neighborhood = {}
    district = {}
    rcl = {}
    i = 0
    while percentage_nodes*len(graph_input.nodes) <= len(unassigned):
        for k in centers_depots:
            district_customers[k]= 0
            district_workload[k] = 0
            district_demand[k] = 0
            neighborhood[k] = []
            neighborhood[k] = near_nodes[k]
            unassigned = unassigned-set(near_nodes[k])-set(centers_depots)
            district[k] = []
            district[k] = Union(district[k], neighborhood[k])
    #Find the total of each activity for each district
            for w in district[k]:
                district_customers[k] = district_customers[k] + graph_input.nodes[w]['n_customers']
                district_workload[k] = district_workload[k] + graph_input.nodes[w]['workload']
                district_demand[k] = district_demand[k] + graph_input.nodes[w]['demand']
                


    local_infeasible = 0

    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(district_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-district_customers[centers_depots[i]],0))+            ((1/average_demand)*max(district_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-district_demand[centers_depots[i]],0))+                ((1/average_workload)*max(district_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-district_workload[centers_depots[i]],0))

    #Select a larger neighborhood for the depots
    larger_selected_nodes = {}
    larger_selected_nodes = {}
    for k in centers_depots:
        larger_selected_nodes[k] = []
        larger_selected_nodes[k] = nx.ego_graph(graph_input,k, radius = 100, center=False, undirected=True, distance='distance')
        larger_selected_nodes[k] = list(set(larger_selected_nodes[k].nodes())-set(district[k]))

    #Ensure that there is no overlap between the neighborhoods
    new_neighborhood = {}
    for k in centers_depots:
        new_neighborhood[k] = []
        for v in larger_selected_nodes[k]:
            x = list(any(v in val for val in new_neighborhood.values()))
            y = list(any(v in val for val in district.values()))
            if True not in x:
                if True not in y:             
                    new_neighborhood[k].append(v)



    #Find the infeasibility of each district
    infeasible = {}

    for k in district:
        infeasible[k] = {}
        for v in new_neighborhood[k]:
            infeasible[k][v] = (1/average_workload)*max(district_workload[k]+graph_input.nodes[v]['workload']-(1+delta)*average_workload,0)+                (1/average_customers)*max(district_customers[k]+graph_input.nodes[v]['n_customers']-(1+delta)*average_customers,0)+                    (1/average_demand)*max(district_demand[k]+graph_input.nodes[v]['demand']-(1+delta)*average_demand,0)

    obj_dispersion = max(shortest_paths_dict[x][y] for i in district for x in district[i] for y in district[i])
    frac_diameter = (1/graph_diameter)
    #Find the average dispersion of each district
    dispersion = {}
    for k in district:
        dispersion[k] = {}
        for v in new_neighborhood[k]:
            dispersion[k][v] = frac_diameter*max(obj_dispersion, max(shortest_paths_dict[x][y] for x in Union(district[k],[v]) for y in Union(district[k],[v])))

    phi = {}

    for k in district:
        phi[k] = {}
        for v in new_neighborhood[k]:
            phi[k][v] = llambda*dispersion[k][v]+(1-llambda)*infeasible[k][v]



    phi_min = {}
    for k in district:
        phi_min[k] = min(phi[k].values())

    phi_max = {}
    for k in district:
        phi_max[k] = max(phi[k].values())



    open_district = {}
    for k in district:
        open_district[k] = True

    #Create the restricted candidate list

    rcl = {}

    for k in district:
        rcl[k] = []
        if open_district[k] == True:
            for h in new_neighborhood[k]:
                if phi[k][h] <= phi_min[k]+rcl_parameter*(phi_max[k]-phi_min[k]):
                    rcl[k].append(h)

    x = 0
    r = 0
    i=0
    viable = False
    OR_OPEN = True
    RCL_EMPTY = True
    NOT_OPEN = False
    UNASSIGNED_REPEAT = False
    final_depot = False
    unassigned_length = len(unassigned)
    unassigned_previous = 0
    while ((len(unassigned) >0) and not NOT_OPEN and not UNASSIGNED_REPEAT):
        if unassigned_length == unassigned_previous:
            UNASSIGNED_REPEAT = True
#         print(unassigned_length)
        unassigned_previous = len(unassigned)
        for k in centers_depots:
            # print("First chosen depot k is")
            # print(k)
            # print("Length of RCL is")
            # print(len(rcl[k]))
            # print("The district is ")
            # print(open_district[k])

            if (len(rcl[k]) == 0):
                #print("RCL EMPTY: Going to next iteration.")
                continue

            if open_district[k]:
                for deleted in rcl[k]:
                    for i in district[k]:
                        if deleted in adjacent_nodes[i]:                        
                            if deleted in rcl[k]:
                                # print("Chosen RCL element is")
                                # print(deleted)
                                rcl[k].remove(deleted)            
                                district[k].append(deleted)
                                district_customers[k] = district_customers[k] + graph_input.nodes[deleted]['n_customers']
                                district_demand[k] = district_demand[k] + graph_input.nodes[deleted]['demand'] 
                                district_workload[k] = district_workload[k] + graph_input.nodes[deleted]['workload'] 
                                #unassigned_previous = len(unassigned)
                                if deleted in unassigned:
                                    unassigned.remove(deleted)
                                    unassigned_length = len(unassigned)
                                if (len(new_neighborhood[k]) <= 0) or (district_customers[k] >= average_customers+delta)                                        or (district_demand[k] >= average_demand+delta) or (district_workload[k] >= average_workload+delta):
                                    open_district[k] = False
            else:
                #print("District closed: Going to next iteration.")
                continue


        # if unassigned_length == unassigned_previous:
        #     UNASSIGNED_REPEAT = True
        # print(unassigned_length)
        # unassigned_previous = len(unassigned)




        if True not in open_district.values():
            NOT_OPEN = True

    #         RCL_EMPTY = False
    #         while len(rcl[depots[r]])<=0 and r<=len(depots):
    #             r = r+1
    #         if r <len(depots):
    #             RCL_EMPTY = True

#     for k in centers_depots:
#         for i in district[k]:
#             print(list(any(i in val for val in district.values())))




    a =  0
    unassigned = list(unassigned)
    for k in centers_depots:
        for x in district[k]:
            for v in unassigned:
                if v in adjacent_nodes[x]:
                    if open_district[k] == True:
                        unique_pls = list(any(v in val for val in district.values()))
                        if True not in unique_pls:
                            district[k].append(v)
                            unassigned.remove(v)
                            district_customers[k] = district_customers[k] + graph_input.nodes[v]['n_customers']
                            district_demand[k] = district_demand[k] + graph_input.nodes[v]['demand'] 
                            district_workload[k] = district_workload[k] + graph_input.nodes[v]['workload'] 
                            if (district_customers[k] >= average_customers+delta) or (district_demand[k] >= average_demand+delta)                                 or (district_workload[k] >= average_workload+delta):
                                open_district[k] = False

                                

    
    a =  0
    unassigned = list(unassigned)
    for k in centers_depots:
        for x in district[k]:
            for v in unassigned:
                if v in adjacent_nodes[x]:
                    unique_pls = list(any(v in val for val in district.values()))
                    if True not in unique_pls:
                        district[k].append(v)
                        unassigned.remove(v)
                        district_customers[k] = district_customers[k] + graph_input.nodes[v]['n_customers']
                        district_demand[k] = district_demand[k] + graph_input.nodes[v]['demand'] 
                        district_workload[k] = district_workload[k] + graph_input.nodes[v]['workload'] 
                        if (district_customers[k] >= average_customers+delta) or (district_demand[k] >= average_demand+delta)                             or (district_workload[k] >= average_workload+delta):
                            open_district[k] = False

    unassigned = list(unassigned)
    #i = depots[0]
    #a = (a+1) % len(depots)
    for k in centers_depots:
        for x in district[k]:
            for v in unassigned:
                if v in adjacent_nodes[x]:
                    unique_pls = list(any(v in val for val in district.values()))
                    if True not in unique_pls:
                        district[k].append(v)
                        unassigned.remove(v)
                        district_customers[k] = district_customers[k] + graph_input.nodes[v]['n_customers']
                        district_demand[k] = district_demand[k] + graph_input.nodes[v]['demand'] 
                        district_workload[k] = district_workload[k] + graph_input.nodes[v]['workload'] 
                        if (district_customers[k] >= average_customers+delta) or (district_demand[k] >= average_demand+delta)                             or (district_workload[k] >= average_workload+delta):
                            open_district[k] = False

    districts_keys = list(district.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in district[districts_keys[k]]:
                color_map[node] = colorss[k]
    

    color_map = list(color_map.values())

#     plt.figure(3,figsize=(12,12))
#     nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
#     plt.show()

    local_infeasible = 0

    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(district_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-district_customers[centers_depots[i]],0))+            ((1/average_demand)*max(district_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-district_demand[centers_depots[i]],0))+                ((1/average_workload)*max(district_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-district_workload[centers_depots[i]],0))

    #print(local_infeasible)
    construction_obj = max(shortest_paths_dict[x][y] for i in district for x in district[i] for y in district[i])
    
    return district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,local_infeasible


# In[316]:


district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,construction_infeasible = construction_grasp(0.05, 0.3,0.7,G1)
for i in district:
    print(len(district[i]))
print(construction_infeasible)
print(construction_obj)


# In[255]:


for k in centers_depots:
    for i in local_sol[k]:
        print(list(any(i in val for val in local_sol.values())))


# In[59]:


#Without Updating Objective Function
    
#Create a function that calculates the linear combination of infeasibility and dispersion for two selected districts.
def localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    district_trial2 = {}
    best_sol = {}

#     print(district_trial2)
#     print(best_sol)
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for i in range(len(centers_depots)):

        moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

    for i in moves:
        moves[i] = list(itertools.chain(*moves[i]))

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    
    while(nmoves<1000 and local_optima==False):
#         print(".....")
#         print(local_optima)
#         print(".....")
        improvement = False
        while((len(moves[k])>0) and (improvement == False)):
            move_to = random.choice(moves[k])
            moves[k].remove(move_to)
            number_of_moves = number_of_moves + 1
#             print("Length of moves is")
#             print(number_of_moves)

#             print(k)
#             print(len(moves[k]))
            for i in district_trial2[k]:
                if move_to in adjacent_nodes[i]:
                    for f in centers_depots:
                        if move_to in district_trial2[f]:
                            district_trial2[f].remove(move_to)

                        unique_districts = list(any(move_to in val for val in district_trial2.values()))
                        if True not in unique_districts:
                            if move_to in adjacent_nodes[i]:
                                district_trial2[k].append(move_to)

#             paths_list = []
#             for nodes in district[k]:
#                 paths_list.append(nx.has_path(graph_input.subgraph(district[k]+[k]),k,nodes))
# #             if False in paths_list:
# #                 for cent in range(len(centers_depots)):
# #                     district_trial2[centers_depots[cent]] = best_sol[centers_depots[cent]][:]
#             print("guess not")
#             if False not in paths_list:
#                 moves[k].remove(move_to)
#     #Use the decision() function to check whether the performed move is an improvement              
            if(decision(best_sol,district_trial2)==True):
#                 print("Yes")
                best_sol = {}
                for i in range(len(centers_depots)):
                    best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]
                nmoves = nmoves+1
#                 print(nmoves)
                #print(decision(best_sol,district_trial2))
                improvement = True
                choose = (choose+1) % p
                kend = k
                k = centers_depots[choose]
                chosen_depots = random.choice(depo_choices[k])
#                     for i in range(len(centers_depots)):
#                         if move_to in moves[centers_depots[i]]:
#                             moves[centers_depots[i]].remove(move_to)
            else:
#                 print("No")
                #print(decision(best_sol,district_trial2))
                for i in range(len(centers_depots)):
                    district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
                #small chance of sad infinite loop :(
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])
#         #         if len(moves[k]) == 0:
#         #             choose = (choose+1) % p
#         #             k = centers_depots[choose]
#         #             chosen_depots = random.choice(depo_choices[k])

        if k == kend:
            local_optima = True
            print("Local Optimum Reached")
#     #     for k in centers_depots:
    #         for i in best_sol[k]:
    #             print(list(any(i in val for val in best_sol.values())))          
#             else:
#                 for i in range(len(centers_depots)):
#                     district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]

    districts_keys = list(best_sol.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in best_sol[districts_keys[k]]:
                color_map[node] = colorss[k]

    color_map = list(color_map.values())

#     plt.figure(3,figsize=(12,12))
#     nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
#     plt.show()
    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[907]:


#Without Updating Objective Function and Remove Adjacency Conditions
    
#Create a function that calculates the linear combination of infeasibility and dispersion for two selected districts.
def localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    district_trial2 = {}
    best_sol = {}

#     print(district_trial2)
#     print(best_sol)
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for i in range(len(centers_depots)):

        moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

    for i in moves:
        moves[i] = list(itertools.chain(*moves[i]))

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    
    while(nmoves<1000 and local_optima==False):
#         print(".....")
#         print(local_optima)
#         print(".....")
        improvement = False
        while((len(moves[k])>0) and (improvement == False)):
            move_to = random.choice(moves[k])
            moves[k].remove(move_to)
            number_of_moves = number_of_moves + 1
#             print("Length of moves is")
#             print(number_of_moves)

#             print(k)
#             print(len(moves[k]))
            for i in district_trial2[k]:
#                 if move_to in adjacent_nodes[i]:
                for f in centers_depots:
                    if move_to in district_trial2[f]:
                        district_trial2[f].remove(move_to)

                unique_districts = list(any(move_to in val for val in district_trial2.values()))
                if True not in unique_districts:
#                         if move_to in adjacent_nodes[i]:
                    district_trial2[k].append(move_to)

#             paths_list = []
#             for nodes in district[k]:
#                 paths_list.append(nx.has_path(graph_input.subgraph(district[k]+[k]),k,nodes))
# #             if False in paths_list:
# #                 for cent in range(len(centers_depots)):
# #                     district_trial2[centers_depots[cent]] = best_sol[centers_depots[cent]][:]
#             print("guess not")
#             if False not in paths_list:
#                 moves[k].remove(move_to)
#     #Use the decision() function to check whether the performed move is an improvement              
            if(decision(best_sol,district_trial2)==True):
#                 print("Yes")
                best_sol = {}
                for i in range(len(centers_depots)):
                    best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]
                nmoves = nmoves+1
#                 print(nmoves)
                #print(decision(best_sol,district_trial2))
                improvement = True
                choose = (choose+1) % p
                kend = k
                k = centers_depots[choose]
                chosen_depots = random.choice(depo_choices[k])
#                     for i in range(len(centers_depots)):
#                         if move_to in moves[centers_depots[i]]:
#                             moves[centers_depots[i]].remove(move_to)
            else:
#                 print("No")
                #print(decision(best_sol,district_trial2))
                for i in range(len(centers_depots)):
                    district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
                #small chance of sad infinite loop :(
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])
#         #         if len(moves[k]) == 0:
#         #             choose = (choose+1) % p
#         #             k = centers_depots[choose]
#         #             chosen_depots = random.choice(depo_choices[k])

        if k == kend:
            local_optima = True
            print("Local Optimum Reached")
#     #     for k in centers_depots:
    #         for i in best_sol[k]:
    #             print(list(any(i in val for val in best_sol.values())))          
#             else:
#                 for i in range(len(centers_depots)):
#                     district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]

    districts_keys = list(best_sol.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in best_sol[districts_keys[k]]:
                color_map[node] = colorss[k]

    color_map = list(color_map.values())

#     plt.figure(3,figsize=(12,12))
#     nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
#     plt.show()
    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[361]:


districts_keys = list(district.keys())
colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

color_map = {}
for node in list(G1.nodes):
    color_map[node] = "blue"
    for k in range(len(districts_keys)):
        if node in district[districts_keys[k]]:
            color_map[node] = colorss[k]

color_map = list(color_map.values())

plt.figure(3,figsize=(12,12))
nx.draw(G1,node_color=color_map, pos=grid_pos,with_labels = True)
plt.show()


# In[976]:


#With Updating Objective Function

#Create a function that calculates the linear combination of infeasibility and dispersion for two selected districts.
def localsearch_grasp(district,construction_obj,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
    def decision(x,y,district_added,district_removed,currentmax,move_to):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']





        
#         weight_district_temp_f_removed = (max(shortest_paths_dict[a][b] for a in y[district_removed] for b in y[district_removed]))
#         weight_district_temp_f_added = (max(shortest_paths_dict[a][move_to] for a in y[district_added]))
        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = currentmax
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True, weight_district_temp_f
        else:

            return False, weight_district_best_f

    district_trial2 = {}
    best_sol = {}

#     print(district_trial2)
#     print(best_sol)
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for i in range(len(centers_depots)):

        moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

    for i in moves:
        moves[i] = list(itertools.chain(*moves[i]))

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    currentmax = construction_obj
    while(nmoves<1000 and local_optima==False):
#         print(".....")
#         print(local_optima)
#         print(".....")
        improvement = False
        while((len(moves[k])>0) and (improvement == False)):
            move_to = random.choice(moves[k])
            moves[k].remove(move_to)
            number_of_moves = number_of_moves + 1
#             print("Length of moves is")
#             print(number_of_moves)
#             print("The current move is", move_to)
#             print(k)
#             print(len(moves[k]))
            for i in district_trial2[k]:
#                 if move_to in adjacent_nodes[i]:
                for f in centers_depots:
                    if move_to in district_trial2[f]:
                        district_trial2[f].remove(move_to)
                        district_removed = f
#                         print("The district removed is" ,f)

                unique_districts = list(any(move_to in val for val in district_trial2.values()))
                if True not in unique_districts:
#                     if move_to in adjacent_nodes[i]:
                    district_trial2[k].append(move_to)
                    district_added = k
#                     print("The district added is", k)

#             paths_list = []
#             for nodes in district[k]:
#                 paths_list.append(nx.has_path(graph_input.subgraph(district[k]+[k]),k,nodes))
# #             if False in paths_list:
# #                 for cent in range(len(centers_depots)):
# #                     district_trial2[centers_depots[cent]] = best_sol[centers_depots[cent]][:]
#             print("guess not")
#             if False not in paths_list:
#                 moves[k].remove(move_to)
#     #Use the decision() function to check whether the performed move is an improvement  
            decision_moment = decision(best_sol,district_trial2,district_added,district_removed,currentmax,move_to)
            if(decision_moment[0]==True):
#                 print("Yes")
                best_sol = {}
                for i in range(len(centers_depots)):
                    best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]
                nmoves = nmoves+1
#                 print(nmoves)
                #print(decision(best_sol,district_trial2))
                improvement = True
                currentmax = decision_moment[1]
                choose = (choose+1) % p
                kend = k
                k = centers_depots[choose]
                chosen_depots = random.choice(depo_choices[k])
#                     for i in range(len(centers_depots)):
#                         if move_to in moves[centers_depots[i]]:
#                             moves[centers_depots[i]].remove(move_to)
            else:
#                 print("No")
                #print(decision(best_sol,district_trial2))
                for i in range(len(centers_depots)):
                    district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
                #small chance of sad infinite loop :(
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])
#         #         if len(moves[k]) == 0:
#         #             choose = (choose+1) % p
#         #             k = centers_depots[choose]
#         #             chosen_depots = random.choice(depo_choices[k])

        if k == kend:
            local_optima = True
            print("Local Optimum Reached")
#     #     for k in centers_depots:
    #         for i in best_sol[k]:
    #             print(list(any(i in val for val in best_sol.values())))          
#             else:
#                 for i in range(len(centers_depots)):
#                     district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]

    districts_keys = list(best_sol.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in best_sol[districts_keys[k]]:
                color_map[node] = colorss[k]

    color_map = list(color_map.values())

#     plt.figure(3,figsize=(12,12))
#     nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
#     plt.show()
    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[389]:


#Without Updating Objective Function and Ordered
    
#Create a function that calculates the linear combination of infeasibility and dispersion for two selected districts.
def localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    district_trial2 = {}
    best_sol = {}


    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for i in range(len(centers_depots)):

        moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

    for i in moves:
        moves[i] = list(itertools.chain(*moves[i]))

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
    for k in moves:
        print(len(moves[k]))

    while(nmoves<1000 and local_optima==False):

        improvement = False
        while((len(moves[k])>0) and (improvement == False)):
#             print(move_index)
            move_to = moves[k][move_index]
            moves[k].remove(move_to)
            number_of_moves = number_of_moves + 1
            
            
            for i in district_trial2[k]:
                if move_to in adjacent_nodes[i]:
                    for f in centers_depots:
                        if move_to in district_trial2[f]:
                            district_trial2[f].remove(move_to)

                        unique_districts = list(any(move_to in val for val in district_trial2.values()))
                        if True not in unique_districts:
                            if move_to in adjacent_nodes[i]:
                                district_trial2[k].append(move_to)
         
            if(decision(best_sol,district_trial2)==True):
#                 print("Yes")
                best_sol = {}
                for i in range(len(centers_depots)):
                    best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]
                nmoves = nmoves+1
                print(nmoves)
                move_index = 0
                improvement = True
                choose = (choose+1) % p
                kend = k
                k = centers_depots[choose]
                chosen_depots = random.choice(depo_choices[k])

            else:
#                 move_index = move_index + 1
                for i in range(len(centers_depots)):
                    district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
                #small chance of sad infinite loop :(
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])


        if k == kend:
            local_optima = True
            print("Local Optimum Reached")



    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[398]:


#28/01/2022

#Without Updating Objective Function and Ordered V.2
    
#Create a function that calculates the linear combination of infeasibility and dispersion for two selected districts.
def localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False
        
        
        
    def merit_function(x):

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

        #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))

        weight_district_best_g = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)

        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

        #Return value of merit function

        return weight_district_best
    

    district_trial2 = {}
    best_sol = {}


    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for i in range(len(centers_depots)):

        moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

    for i in moves:
        moves[i] = list(itertools.chain(*moves[i]))

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
    for k in moves:
        print(len(moves[k]))
        
        
    node_district_matching = {}

    for k in district:
        for i in graph_input.nodes:
            if i in district[k]:
                node_district_matching[i] = k
            
    current_best_solution = merit_function(district)

    while(nmoves<1000 and local_optima==False):

        improvement = False
        while((len(moves[k])>0) and (improvement == False)):
#             print(move_index)
            move_to = moves[k][move_index]
            moves[k].remove(move_to)
            number_of_moves = number_of_moves + 1
            
            district_trial2[node_district_matching[move_to]].remove(move_to)
            district_trial2[k].append(move_to)
            
            
#             for i in district_trial2[k]:
#                 if move_to in adjacent_nodes[i]:
#                     for f in centers_depots:
#                         if move_to in district_trial2[f]:
#                             district_trial2[f].remove(move_to)

#                         unique_districts = list(any(move_to in val for val in district_trial2.values()))
#                         if True not in unique_districts:
#                             if move_to in adjacent_nodes[i]:
#                                 district_trial2[k].append(move_to)

            testing_solution = merit_function(district_trial2)
    
    
            if(testing_solution < current_best_solution):
#                 print("Yes")
                best_sol = {}
                for i in range(len(centers_depots)):
                    best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]
                current_best_solution = testing_solution
                node_district_matching = {}

                for location in best_sol:
                    for bu in graph_input.nodes:
                        if bu in best_sol[location]:
                            node_district_matching[bu] = location
                            
                nmoves = nmoves+1
                print(nmoves)
                move_index = 0
                improvement = True
                choose = (choose+1) % p
                kend = k
                k = centers_depots[choose]
                chosen_depots = random.choice(depo_choices[k])
                
#                 moves = {}

#                 for i in range(len(centers_depots)):

#                     moves[centers_depots[i]]= [v for k,v in best_sol.items() if k != centers_depots[i]]

#                 for i in moves:
#                     moves[i] = list(itertools.chain(*moves[i]))

            else:
#                 move_index = move_index + 1
                for i in range(len(centers_depots)):
                    district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
                #small chance of sad infinite loop :(
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])


        if k == kend:
            local_optima = True
            print("Local Optimum Reached")



    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[ ]:


#Add a dictionary with the keys as nodes. Value of each key to be the district they are in. 
#Update each time a move is performed.


# In[348]:


len(node_district_matching)


# In[346]:


node_district_matching = {}

for k in district:
    for i in G1.nodes:
        if i in district[k]:
            node_district_matching[i] = k


# In[357]:


node_district_matching


# In[353]:


centers_depots


# In[356]:


district[node_district_matching[651]]


# In[352]:


testing_stuff = {1:[2,3,4,5],6:[7,8,9,10]}

testing_stuff[6]+[12]


# In[351]:


testing_stuff


# In[358]:


#Without Updating Objective Function and Ordered Best Improvement V.2

#Check all nodes and pick the best
    
#Create a function that calculates the linear combination of infeasibility and dispersion for two selected districts.
def localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
   
    #Save value of best solution for comparisons. Onlly calculate the new one. Update when you find a better solution.
    #Update membership of each district. 
    
    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    
    def merit_function(x):

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))

        weight_district_best_g = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)

        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return value of merit function

        return weight_district_best
        
        
        
    district_trial2 = {}
    best_sol = {}

    
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for i in range(len(centers_depots)):

        moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

    for i in moves:
        moves[i] = list(itertools.chain(*moves[i]))

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
    for k in moves:
        print(len(moves[k]))

        
    node_district_matching = {}

    for k in district:
        for i in graph_input.nodes:
            if i in district[k]:
                node_district_matching[i] = k
            
    current_best_solution = merit_function(district)
    
    
    while(nmoves<1000 and local_optima==False):

        improvement = False
        while(improvement == False):
            solution_objective = []
            solution_allocation = []
            number_of_moves = number_of_moves + 1
            for basic_unit in moves[k]:
                previous_center = node_district_matching[basic_unit]
                district_trial2[node_district_matching[basic_unit]].remove(basic_unit)
                district_trial2[k].append(basic_unit)
                new_objective = merit_function(district_trial2)
                solution_objective.append(new_objective)
                solution_allocation.append(district_trial2)
                district_trial2[k].remove(basic_unit)
                district_trial2[node_district_matching[basic_unit]].append(basic_unit)
            
#             print(move_index)
#             move_to = moves[k][move_index]
#             moves[k].remove(move_to)
            best_value = min(solution_objective)
            best_value_index = solution_objective.index(best_value)
            district_trial2 = cp.deepcopy(solution_allocation[best_value_index])    
            
            
            for i in district_trial2[k]:
                if move_to in adjacent_nodes[i]:
                    #Just remove directly.
                    for f in centers_depots:
                        if move_to in district_trial2[f]:
                            district_trial2[f].remove(move_to)

                        unique_districts = list(any(move_to in val for val in district_trial2.values()))
                        if True not in unique_districts:
                            if move_to in adjacent_nodes[i]:
                                district_trial2[k].append(move_to)
           
#             merit_function()  
#             if new merit_function()<best_sol:
#                 update best solution
#             490
#             []
            
        
            if(decision(best_sol,district_trial2)==True):
#                 print("Yes")
                best_sol = {}
                for i in range(len(centers_depots)):
                    best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]
                nmoves = nmoves+1
                print(nmoves)
                move_index = 0
                improvement = True
                choose = (choose+1) % p
                kend = k
                k = centers_depots[choose]
#                 chosen_depots = random.choice(depo_choices[k])

            else:
#                 move_index = move_index + 1
                for i in range(len(centers_depots)):
                    district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
                #small chance of sad infinite loop :(            
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])


        if k == kend:
            local_optima = True
            print("Local Optimum Reached")



    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[ ]:


#Check which node improves the district the most (For the first improvement variant)
#Stop at the first district that improves the solution (For the best improvement variant)


# In[335]:


#Without Updating Objective Function and Ordered and Best Improvement
    
#Create a function that calculates the linear combination of infeasibility and dispersion for two selected districts.
def localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False


    def merit_function(x):

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)

        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return value of merit function

        return weight_district_best

        
        
        
    district_trial2 = {}
    best_sol = {}


    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for k in centers_depots:
        moves[k] = cp.deepcopy(district[k])
        
    

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
    for k in moves:
        print(len(moves[k]))

    while(nmoves<1000 and local_optima==False):
        
        improvement = False
        while((len(moves[k])>0) and (improvement == False)):
            
#             print(move_index)
#             move_from = moves[k][move_index]
     
            move_from = random.choice(moves[k])
            moves[k].remove(move_from)
            number_of_moves = number_of_moves + 1
#             print(move_from)
            district_trial2[k].remove(move_from)
            solution_objective = []
            solution_allocation = []
            districting_old = cp.deepcopy(district_trial2)
            # for center_to except k do
            for center_to in district_trial2:
                if center_to == k: continue
                district_trial2[center_to].append(move_from)
#                 sol_obj = (max(shortest_paths_dict[a][b] for i in district_trial2 for a in district_trial2[i] for b in district_trial2[i]))
                sol_obj = merit_function(district_trial2)
                sol_alloc = cp.deepcopy(district_trial2)
                solution_allocation.append(cp.deepcopy(sol_alloc))
                solution_objective.append(sol_obj)
                district_trial2 = cp.deepcopy(districting_old)
                
                
                
            best_value = min(solution_objective)
            best_value_index = solution_objective.index(best_value)
            district_trial2 = cp.deepcopy(solution_allocation[best_value_index])            
#             for i in solution_allocation[6]:
#                 if move_from in solution_allocation[6][i]:
#                     print(i)
            if(decision(best_sol,district_trial2)==True):
                print(k)
                print("Yes")
                best_sol = {}
                for i in range(len(centers_depots)):
                    best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]
                nmoves = nmoves+1
#                 print(nmoves)
                move_index = 0
                improvement = True
                choose = (choose+1) % p
                kend = k
                k = centers_depots[choose]
                chosen_depots = random.choice(depo_choices[k])
                for k in centers_depots:
                    moves[k] = cp.deepcopy(best_sol[k])
            else:
#                 move_index = move_index + 1
                for i in range(len(centers_depots)):
                    district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
#                 for k in centers_depots:
#                     moves[k] = cp.deepcopy(districting_old[k])
                #small chance of sad infinite loop :(
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])


        if k == kend:
            local_optima = True
            print("Local Optimum Reached")



    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[385]:


#Without Updating Objective Function and Ordered V.2

#Check all nodes and pick the best
    
#Create a function that calculates the linear combination of infeasibility and dispersion for two selected districts.
def localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
   
    #Save value of best solution for comparisons. Onlly calculate the new one. Update when you find a better solution.
    #Update membership of each district. 
    
    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    
    def merit_function(x):

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))

        weight_district_best_g = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)

        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return value of merit function

        return weight_district_best
        
    districts_keys = list(district.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in district[districts_keys[k]]:
                color_map[node] = colorss[k]


    color_map = list(color_map.values())

    plt.figure(3,figsize=(12,12))
    nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
    plt.show()
        
    district_trial2 = {}
    best_sol = {}

    
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for i in range(len(centers_depots)):

        moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

    for i in moves:
        moves[i] = list(itertools.chain(*moves[i]))

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
    # for k in moves:
    #     print(len(moves[k]))

        
    node_district_matching = {}

    for k in district:
        for i in graph_input.nodes:
            if i in district[k]:
                node_district_matching[i] = k
            
    current_best_solution = merit_function(district)
    # print(current_best_solution)
    
    while(nmoves<1000 and local_optima==False):

        improvement = False
#         while(improvement == False):
        solution_objective = []
        solution_allocation = []
        number_of_moves = number_of_moves + 1
        for basic_unit in moves[k]:
            # previous_center = node_district_matching[basic_unit]
            district_trial2[node_district_matching[basic_unit]].remove(basic_unit)
            district_trial2[k].append(basic_unit)
            new_objective = merit_function(district_trial2)
            solution_objective.append(new_objective)
            solution_allocation.append(cp.deepcopy(district_trial2))
            district_trial2[k].remove(basic_unit)
            district_trial2[node_district_matching[basic_unit]].append(basic_unit)
            # print(basic_unit)


        best_value = min(solution_objective)
        best_value_index = solution_objective.index(best_value)
        district_trial2 = cp.deepcopy(solution_allocation[best_value_index])    
#             print(best_value)


        if(best_value<current_best_solution):

#                 print("Yes")
            best_sol = cp.deepcopy(solution_allocation[best_value_index]) 
#             for i in range(len(centers_depots)):
#                 best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]

            districts_keys = list(best_sol.keys())
            colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

            color_map = {}
            for node in list(graph_input.nodes):
                color_map[node] = "blue"
                for locations in range(len(districts_keys)):
                    if node in best_sol[districts_keys[locations]]:
                        color_map[node] = colorss[locations]


            color_map = list(color_map.values())

            plt.figure(3,figsize=(12,12))

            nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
            plt.show()

            nmoves = nmoves+1
            print(nmoves)
            move_index = 0
            improvement = True
            choose = (choose+1) % 9
            kend = k
            k = centers_depots[choose]
#                 chosen_depots = random.choice(depo_choices[k])
            print("The k and kend are:")
            print(k)
            print(kend)
            node_district_matching = {}

            for centers in best_sol:
                for nodes in graph_input.nodes:
                    if nodes in best_sol[centers]:
                        node_district_matching[nodes] = centers

            # print(current_best_solution)        
            current_best_solution = best_value
            print("The best solution so far:")
            print(current_best_solution)
            moves = {}

            for i in range(len(centers_depots)):

                moves[centers_depots[i]]= [v for k,v in best_sol.items() if k != centers_depots[i]]

            for i in moves:
                moves[i] = list(itertools.chain(*moves[i]))

        else:
            print("We failed so we came here")
            district_trial2 = cp.deepcopy(best_sol)
#                 move_index = move_index + 1
            for i in range(len(centers_depots)):
                district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
            improvement = False


        print("I came here boi")
            #small chance of sad infinite loop :(            
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])
        print(k)
        print("---------------------------")

        if k == kend:
            local_optima = True
            print("Local Optimum Reached")



    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[280]:


def shaking(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input,shaking_steps):    
    
    district_trial2 = {}
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

   
    for depot in centers_depots:
        for step in range(shaking_steps):
            node_to_move = random.choice(district_trial2[depot])
            district_to_receive = random.choice(list(set(centers_depots)-set([depot])))
            district_trial2[district_to_receive].append(node_to_move)
            district_trial2[depot].remove(node_to_move)
    
    best_sol = cp.deepcopy(district_trial2)

    districts_keys = list(best_sol.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in best_sol[districts_keys[k]]:
                color_map[node] = colorss[k]

    color_map = list(color_map.values())

    plt.figure(3,figsize=(12,12))
    nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
    plt.show()
    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[289]:


#shaking with only one district
def shaking(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input,shaking_steps):    
    
    district_trial2 = {}
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    selections = cp.deepcopy(centers_depots)
    depot_from = random.choice(selections)
    selections.remove(depot_from)
    depot_to = random.choice(selections)

    for step in range(shaking_steps):
        node_to_move = random.choice(district_trial2[depot_from])
        district_trial2[depot_to].append(node_to_move)
        district_trial2[depot_from].remove(node_to_move)

    best_sol = cp.deepcopy(district_trial2)

    districts_keys = list(best_sol.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in best_sol[districts_keys[k]]:
                color_map[node] = colorss[k]

    color_map = list(color_map.values())

    plt.figure(3,figsize=(12,12))
    nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
    plt.show()
    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


# In[74]:


district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,construction_infeasible = construction_grasp(0.05, 0.3,0.7,G1)


# In[46]:


get_ipython().run_line_magic('timeit', 'local_obj, local_infeasible, local_sol = localsearch_grasp(district,construction_obj,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)')


# In[386]:


#Best Improvement Local Search
local_obj, local_infeasible, local_sol = localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)


# In[390]:


#First Improvement Local Search
local_obj, local_infeasible, local_sol = localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)


# In[399]:


#28/01/2022
#First Improvement Local Search (edited)
local_obj, local_infeasible, local_sol = localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)


# In[158]:


centers_depots


# In[387]:


print(construction_obj)
print(construction_infeasible)


# In[673]:


0.7*construction_obj+0.3*construction_infeasible


# In[ ]:


424.9158018113799


# In[624]:


print(shaking_obj)
print(shaking_infeasible)


# In[396]:


#28/01/2022
#First Improvement Local Search (edited)
print(local_obj)
print(local_infeasible)


# In[400]:


#28/01/2022
#First Improvement Local Search (edited), without updating moves
print(local_obj)
print(local_infeasible)


# In[388]:


#Best Improvement Local Search
print(local_obj)
print(local_infeasible)


# In[391]:


#First Improvement Local Search
print(local_obj)
print(local_infeasible)


# In[777]:


local_obj, local_infeasible, local_sol = localsearch_grasp(local_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
local_obj, local_infeasible, local_sol = localsearch_grasp(local_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
local_obj, local_infeasible, local_sol = localsearch_grasp(local_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
local_obj, local_infeasible, local_sol = localsearch_grasp(local_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
local_obj, local_infeasible, local_sol = localsearch_grasp(local_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)


# In[796]:


brbrhehe = {1: [284.06636315360134, 284.06636315360134],
 2: [462.0108290786542, 462.0108290786542],
 4: [350.7662135282603, 350.7662135282603],
 5: [261.73202685290124, 261.73202685290124],
 6: [363.4360323415545, 359.7039604197904],
 7: [357.3387883015249, 357.3387883015249],
 8: [337.80497768452545, 311.83347039234275],
 9: [295.93893698124685, 295.93893698124685],
 10: [443.3177231917283, 415.8940340487734],
 11: [332.79598324911325, 332.79598324911325],
 12: [393.49271216653307, 371.0528041858879],
 13: [446.0146422079629, 388.7103834231823],
 14: [499.27297475686413, 499.27297475686413],
 15: [357.2201684913621, 357.2201684913621],
 16: [314.48427157266485, 314.48427157266485],
 17: [396.55616942462143, 316.3054667046266]}


# In[ ]:


brbrhehe2 = {}


# In[799]:


pd.DataFrame.from_dict(brbrhehe).transpose(copy=False)


# In[ ]:


fig = sns.barplot(x='Instance', y='Percentage Improvement', hue='type', data=df).set_title("Percentage Improvement Between Construction & Local Search")


# In[745]:


results_vns4


# In[801]:


[0.7*432+0.3*6.312706860591748, 0.7*649+0.3*0.07118570931622]


# In[802]:


[0.7*520+0.3*9.62283151849268, 0.7*668+0.3*0.36446338223474156]


# In[791]:


results_vns4


# In[795]:


results_vns5


# In[826]:


results_vns6 = {}


# In[825]:


results_vns6


# In[822]:


results_vns6


# In[819]:


results_vns6


# 1) Do "not VNS" on only one construction.
# 2) Work on an updating function for diameter.

# In[828]:


results_vns6


# In[892]:


district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,construction_infeasible = construction_grasp(0.05, 0.3,0.7,G1)
print(construction_obj)


# In[893]:


for i in district:
    print(len(district[i]))


# In[989]:


results_vns6 = []


# In[992]:


results_vns6


# In[1072]:


algorithm3 = []


# In[1075]:


algorithm3


# In[1074]:


for instance in range(10):
    local_obj, local_infeasible, local_sol = localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
    algorithm3.append((local_obj,local_infeasible))


# In[207]:


algorithm2 = {}


# In[198]:


algorithm2


# In[169]:


algorithm2


# In[88]:


algorithm2


# In[1071]:


algorithm2


# In[83]:


district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,construction_infeasible = construction_grasp(0.05, 0.3,0.7,G1)
for i in district:
    print(len(district[i]))
print(construction_infeasible)
print(construction_obj)


# In[239]:


centers_depots


# In[291]:


algorithm2


# In[312]:


centers_depots


# In[397]:


districts_keys = list(local_sol.keys())
colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

color_map = {}
for node in list(G1.nodes):
    color_map[node] = "blue"
    for k in range(len(districts_keys)):
        if node in local_sol[districts_keys[k]]:
            color_map[node] = colorss[k]


color_map = list(color_map.values())

plt.figure(3,figsize=(12,12))
nx.draw(G1,node_color=color_map, pos=grid_pos,with_labels = True)
plt.show()


# In[290]:


for instance in range(1):
    algorithm2[instance] = []
    incumbant_solution = cp.deepcopy(district)
    current = 0.7*construction_obj+0.3*construction_infeasible
    algorithm2[instance].append(construction_obj)
    algorithm2[instance].append(construction_infeasible)
    failure = 0

    while failure <3:
        k = 1
        while k <= 5:
            print("-----------------------")
            print("current neighborhood is")
            print(k)
            print("-----------------------")
            shaking_obj, shaking_infeasible, shaking_sol = shaking(incumbant_solution,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1,k)
            local_obj, local_infeasible, local_sol = localsearch_grasp(shaking_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
            new = 0.7*local_obj+0.3*local_infeasible
            if new<current:
                best_solution = cp.deepcopy(local_sol)
                algorithm2[instance].append(local_obj)
                algorithm2[instance].append(local_infeasible)
                k = 1
                current = cp.deepcopy(new)
                incumbant_solution = local_sol
            else:
                k = k+1
    #             incumbant_solution = local_sol
                if k == 5:
                    failure = failure +1
                    print("We failed boi", failure)

    algorithm2[instance].append(local_obj)
    algorithm2[instance].append(local_infeasible)


# In[127]:


local_obj, local_infeasible, local_sol = localsearch_grasp(shaking_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)


# In[128]:


local_obj


# In[129]:


local_infeasible


# In[124]:


#This is with ordered Local Search
algorithm2


# In[96]:


#This is with random Local Search
algorithm2


# In[1063]:


algorithm1 = {}


# In[1066]:


algorithm1


# In[ ]:


for instance in range(10):
    algorithm1[instance] = []
    print("***********************")
    print("Performing Instance")
    print(instance)
    print("***********************")
    k = 1
    best_solution = cp.deepcopy(district)
    current = 0.7*construction_obj+0.3*construction_infeasible
    algorithm1[instance].append(construction_obj)
    algorithm1[instance].append(construction_infeasible)
    while k <= 3:
        print("-----------------------")
        print("current neighborhood is")
        print(k)
        print("-----------------------")
        shaking_obj, shaking_infeasible, shaking_sol = shaking(best_solution,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1,k)
        local_obj, local_infeasible, local_sol = localsearch_grasp(shaking_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
        local_obj, local_infeasible, local_sol = localsearch_grasp(local_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
        local_obj, local_infeasible, local_sol = localsearch_grasp(local_sol,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
        new = 0.7*local_obj+0.3*local_infeasible
        
        if new<current:
            best_solution = cp.deepcopy(local_sol)
            algorithm1[instance].append(local_obj)
            algorithm1[instance].append(local_infeasible)
            k = 1
            current = cp.deepcopy(new)
            print(new)
        else:
            k = k+1

algorithm1[instance].append((local_obj,local_infeasible))


# In[24]:


@dataclass
class Best_Solutions:
    id : uuid.UUID
    obj : int
    inf : float
    RDB: float
    imp: float
    allocation : dict

    def __lt__(self,other):
        return self.obj<other.obj

sorted_elite_solutions = []


# In[388]:


trial_solutions = {}
for i in range(len((sorted_elite_solutions))):
    trial_solutions[i] = sorted_elite_solutions[i].allocation


# In[393]:


elite_solutions = {}
for i in range(len((sorted_elite_solutions))):
    elite_solutions[i] = sorted_elite_solutions[i].allocation


# In[ ]:


sorted_trial_solutions = []
for i in range(20):
    try:
        district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,local_infeasible_construction = construction_grasp(0.05, 0.3,0.7,G1)
        best_obj, local_infeasible_lc,best_sol = localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)
        print(i)
        bisect.insort_left(sorted_trial_solutions, Best_Solutions(uuid.uuid1(), best_obj, local_infeasible_lc, ((construction_obj-best_obj)/construction_obj)*100,((local_infeasible_construction-local_infeasible_lc)/local_infeasible_construction)*100,best_sol))
    except:
        print("Did not work here")
        pass


# In[173]:


from networkx.algorithms import bipartite
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
from networkx.algorithms.bipartite import sets as bipartite_sets
from networkx.algorithms.bipartite import minimum_weight_full_matching


# In[163]:


brr = list(sorted_elite_solutions[0].allocation.keys())
brr2 = list(sorted_elite_solutions[1].allocation.keys())


# In[165]:


removed = []
for i in brr:
    if i in brr2:
        removed.append(i)
        brr.remove(i)
        brr2.remove(i)


# In[170]:


B = nx.Graph()
# Add nodes with the node attribute "bipartite"
B.add_nodes_from(brr, bipartite=0)
B.add_nodes_from(brr2, bipartite=1)
B.add_edges_from(centers_comb)

#brr = list(sorted_elite_solutions[0].allocation.keys())


# In[198]:


for i in brr:
    if i in sorted_elite_solutions[2].allocation.keys():
        print(i)


# In[171]:


nx.set_edge_attributes(B, values = centers_shortest, name = "distance")


# In[166]:


centers_comb = list(itertools.product(brr, brr2))


# In[167]:


comb_lengths = []
for i in brr:
    for x in brr2:
        comb_lengths.append((shortest_paths_dict[x][i]))


# In[354]:


for i in brr:
    if i in brr2:
        print(i)


# In[168]:


centers_shortest = dict(zip(centers_comb, comb_lengths))


# In[174]:


match2 = minimum_weight_full_matching(B, top_nodes=brr, weight='distance')


# In[175]:


g_match2=nx.Graph()
for kk,vv in match2.items():
    g_match2.add_edge(kk,vv)


# In[253]:


a = [elem for elem in brr2 if elem not in brr ]


# In[254]:


b = [elem for elem in brr if elem not in brr2 ]


# In[244]:


for i in range(len(brr)):
    for j in range(len(brr2)):
        if brr[i] == brr[j]:
            print(i)


# In[238]:


str(20)+"_a"


# In[77]:


get_ipython().run_line_magic('lprun', '-f localsearch_grasp localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)')


# In[180]:


centers_matchings = [(k, v) for k, v in match2.items()]


# In[194]:


centers_matchings


# In[300]:


temp = set(centers_matchings) & {(b, a) for a, b in centers_matchings}
centers_matchings = {(a, b) for a, b in temp if a < b}


# In[188]:


centers_matchings = centers_matchings[:int(len(centers_matchings)/2)]


# In[193]:


centers_matchings.append((removed[0],removed[0]))


# In[145]:


plt.figure(figsize=(13, 7))

fig = sns.barplot(x='Instance', y='Percentage Improvement', hue='type', data=df).set_title("Percentage Improvement Between Construction & Local Search")
#fig.title("Percentage Improvement Between Construction & Local Search")


# In[31]:


district.keys()


# In[125]:


districts_keys = list(district.keys())

colorss = ["darkorange","lawngreen", "maroon"]

color_map = {}
for node in list(G1.nodes):
    color_map[node] = "blue"
    for k in range(len(districts_keys)):
        if node in district[districts_keys[k]]:
            color_map[node] = colorss[k]

color_map = list(color_map.values())

plt.figure(3,figsize=(12,12))
nx.draw(G1,node_color=color_map, pos=grid_pos,with_labels = True)
plt.show()


# In[37]:


local_obj, local_infeasible, local_sol = localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,0.3,0.05,G1)


# In[497]:


G2 = nx.Graph()
G2.add_nodes_from(list(trial_districts.keys()))
G2.add_edge(34,81)
G2.add_edge(34,258)
G2.add_edge(34,707)
G2.add_edge(34,359)
G2.add_edge(258,269)
G2.add_edge(258,134)
G2.add_edge(258,359)
G2.add_edge(656,707)
G2.add_edge(656,359)
G2.add_edge(656,668)
G2.add_edge(707,359)
G2.add_edge(668,511)
G2.add_edge(668,269)
G2.add_edge(134,269)
G2.add_edge(511,269)


# In[480]:


positions = {}
for i in G2.nodes:
    if i in grid_pos:
        positions[i]=grid_pos[i]
#grid_pos


# In[ ]:


[[34, 359, 258],
 [34, 707, 359],
 [656, 707, 359],
 [269, 668, 656, 359, 258],
 [269, 511, 668],
 [269, 134, 258]]


# In[500]:


karate = nx.generators.social.karate_club_graph()
communities = list(nx.community.asyn_fluidc(karate, 4))


# In[501]:


communities


# In[498]:


nx.cycle_basis(G2, root=None)


# In[475]:


list(trial_districts.keys())


# In[236]:


nodegobrr


# In[470]:


trial_districts = cp.deepcopy(district)


# In[469]:


trail_districts2 = cp.deepcopy(trial_districts)


# In[432]:


int(floor(0.7*len(trial_districts[34])))


# In[ ]:


def split_district(district,percentage)
    starting_node = random.choice(district)
    partial_district = [starting_node]
    while len(partial_district) < int(floor(percentage*len(district))):
        chosen_node = random.choice(district)
        if nx.has_path(G1.subgraph(partial_district))


# In[464]:


nodegobrr= [157]
trial_districts[134].remove(157)
while len(nodegobrr) < int(floor(0.4*len(trial_districts[134]))):
    chosen_node = random.choice(trial_districts[134])
    if nx.has_path(G1.subgraph(nodegobrr+[chosen_node]),nodegobrr[0],chosen_node) == True:
        nodegobrr.append(chosen_node)
        trial_districts[134].remove(chosen_node)
        paths_list = []
        for items in trial_districts[134]:
            paths_list.append(nx.has_path(G1.subgraph(trial_districts[134]+[134]),134,items))
        if False in paths_list:
            trial_districts[134].append(chosen_node)
            nodegobrr.remove(chosen_node)


# In[438]:


nodegobrr= [157]
trial_districts[134].remove(157)
while len(nodegobrr) < int(floor(0.4*len(trial_districts[34]))):
    for node in trial_districts[134]:
        for i in nodegobrr:
            if node in adjacent_nodes[i]:
                if node not in nodegobrr:
#                     print("okay?")
                    nodegobrr.append(node)
                    trial_districts[134].remove(node)
                    paths_list = []
                    for items in trial_districts[134]:
                        paths_list.append(nx.has_path(G1.subgraph(trial_districts[134]+[134]),134,items))
                    if False in paths_list:
#                         print("lol")
                        trial_districts[134].append(node)
                        nodegobrr.remove(node)
                    else:
                        print("worked")

