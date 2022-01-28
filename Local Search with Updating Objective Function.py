#Third Local Search

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
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol
