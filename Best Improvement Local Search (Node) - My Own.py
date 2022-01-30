#30/01/2022

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
        
        
    
    
    district_trial2 = {}
    best_sol = {}

    
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]
        
    node_district_matching = {}
    for k in district:
        for i in district[k]:
            node_district_matching[i] = k

#     #Initialize a dictionary of moves
#     moves = {}

#     for i in range(len(centers_depots)):

#         moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

#     for i in moves:
#         moves[i] = list(itertools.chain(*moves[i]))
     
    moves = {}
    for depots in centers_depots:
        moves[depots] = []
        for nodes in node_district_matching:
            if node_district_matching[nodes] != depots:
                moves[depots].append(nodes)

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

        

    
    
#     highest = 0
#     for i in district:
#         for x in district[i]:
#             if max(shortest_paths_dict[x][y]for y in district[i])>highest:
#                 highest = max(shortest_paths_dict[x][y] for y in district[i])
#                 highest_depot = i
    
            
    current_best_objective = merit_function(district)
    print(current_best_objective)
    # print(current_best_solution)
    current_best_solution = cp.deepcopy(district)
    trial_solution = cp.deepcopy(district)
    trial_objective = current_best_objective
    while(nmoves<1000 and local_optima==False):
        solution_objective = []
        solution_allocation = []
        number_of_moves = number_of_moves + 1
        improvement = False
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
#         print(best_value)

        districts_keys = list(district_trial2.keys())
        colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

        color_map = {}
        for node in list(graph_input.nodes):
            color_map[node] = "blue"
            for locations in range(len(districts_keys)):
                if node in district_trial2[districts_keys[locations]]:
                    color_map[node] = colorss[locations]


        color_map = list(color_map.values())

        plt.figure(3,figsize=(12,12))
        nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
        plt.show()


        if(best_value<current_best_objective):

            print("Yes")
            best_sol = cp.deepcopy(solution_allocation[best_value_index]) 
#             for i in range(len(centers_depots)):
#                 best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]



            print(nmoves)
            move_index = 0
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
            current_best_objective = best_value
            print("The best solution so far:")
            print(current_best_objective)
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
            choose = (choose+1) % 9
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
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol
