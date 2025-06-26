###################################################################
#### part of SAILoR (Structure-Aware Inference of Logic Rules) ####
################################################################### 

#Copyright 2023 Žiga Pušnik   

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import networkx as nx 
import numpy as np  

#global variable 
tri_types = [ 1, 2, 2, 4, 2, 3, 6, 8, 2, 6, 5, 9, 4, 8, 9, 13, 2, 6, 3, 8, 5, 7, 7, 11, 6, 10, 7, 14, 9, 14, 12, 15, 2, 5, 6, 9, 6, 7, 10, 14, 3, 7, 7, 12, 8, 11, 14, 15, 4, 9, 8, 13, 9, 12, 14, 15, 8, 14, 11, 15, 13, 15,15, 16]   

#modified triadic census algorithm for orbit count inspired by Ortmann and Brandes, 2017 
#based on Batagelj and Mrvar, 2000  
#calculates triadic census of a simple directed graph    
def count_triads(G):   
    #G ... networkx directed graph (digraph)     
    #nodelist ... list of considered nodes, if not provided, consider all nodes in G

        
    """
    tri_type_orbit_map = [[0,0,0],[2,3,1],[2,1,3],[11, 12, 12],[3,2,1],[5,5,4],[6,7,8],[14,15,13],[1,2,3],[7,6,8],[10,10,9],[23,24,22],[12,11,12],[15,14,13],[24,23,22],[26,26,25],
                          [3,1,2],[6,8,7],[5,4,5],[14,13,15],[9,10,10],[17,18,16],[17,16,18],[20,19,19],[8,7,6],[21,21,21],[18,16,17],[30,29,31],[22,23,24],[31,30,29],[28,27,28],[33,32,34],
                          [1,3,2],[10,9,10],[7,8,6],[23,22,24],[8,6,7],[18,17,16],[21,21,21],[30,31,29],[4,5,5],[16,17,18],[16,18,17],[27,28,28],[13,14,15],[19,20,19],[29,30,31],[32,33,34],
                          [12,12,11],[24,22,23],[15,13,14],[26,25,26],[22,24,23],[28,28,27],[31,29,30],[33,34,32],[13,15,14],[29,31,30],[19,19,20],[32,34,33],[25,26,26],[34,33,32],[34,32,33],[35,35,35]]

    orbits_to_triads = np.array([1, 2, 2, 2, 3, 3, 6, 6, 6, 5, 5, 4, 4, 8, 8, 8, 7, 7, 7, 11, 11, 10, 9, 9, 9, 13, 13, 12, 12, 14, 14, 14, 15, 15, 15, 16]) - 1    
    triads_to_orbits = [[0],[1, 2, 3],[4,5],[11,12],[9,10],[6,7,8],[16,17,18],[13,14,15],[22,23,24],[21],[19,20],[27,28],[25,26],[29,30,31],[32,33,34],[35]]     
    """    

    #initialize empty triadic census for triads #["021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]  
    census = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])            

    #initialize empty orbit count for triadic census 
    triad_pair_count = np.zeros((G.number_of_nodes(), G.number_of_nodes(), 16))              

    """ 
    if nodelist: 
        G = nx.subgraph(G, nodelist)       
    """   
        
    nodelist = list(G.nodes)         

    for v in nodelist: 
        R_v = set(nx.all_neighbors(G, v)) 

        for u in R_v: 
            if v >= u: 
                continue 

            R_u = set(nx.all_neighbors(G, u))   
            S = R_v.union(R_u)  

            for w in S: 
                if u >= w and (v >= w or w in R_v):    
                    continue

                #update triadic census
                tri_code = int(G.has_edge(v,u))  + 2*int(G.has_edge(v, w)) + 4*int(G.has_edge(u, v)) + 8*int(G.has_edge(u,w)) + 16*int(G.has_edge(w,v)) + 32*int(G.has_edge(w,u)) 
                tri_type = tri_types[tri_code]-1
                census[tri_type] = census[tri_type] + 1 
               
                triad_pair_count[u,v, tri_type] = triad_pair_count[u,v, tri_type] + 1       
                triad_pair_count[u,w, tri_type] = triad_pair_count[u,w, tri_type] + 1   
                triad_pair_count[v,w, tri_type] = triad_pair_count[v,w, tri_type] + 1      

                #update orbit count 
                #o_v, o_u, o_w  = tri_type_orbit_map[tri_code]    
                #orbit_count[v, o_v] = orbit_count[v, o_v] + 1  
                #orbit_count[u, o_u] = orbit_count[u, o_u] + 1  
                #orbit_count[w, o_w] = orbit_count[w, o_w] + 1    

    #ignore first three triads with multiple connected components           
    return census[3:], triad_pair_count     


#if u and v are connected subgraphs nodes is union of all neighbours else subgraph nodes is intersect of all neighbours 
#intersect and union operation is handled prior to calling count_local_triads
def count_local_triads(G, v, u, subgraph_nodes):
    triad_pair_count = np.zeros(16)  

    for a in subgraph_nodes: 
        tri_code = int(G.has_edge(v,u))  + 2*int(G.has_edge(v, a)) + 4*int(G.has_edge(u, v)) + 8*int(G.has_edge(u,a)) + 16*int(G.has_edge(a,v)) + 32*int(G.has_edge(a,u))   
        tri_type = tri_types[tri_code] - 1    
    
        triad_pair_count[tri_type] = triad_pair_count[tri_type] + 1                  

    return triad_pair_count         