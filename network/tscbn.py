#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

import random
import copy
import numpy as np
from _include.m_libpgm.tablecpdfactorization import TableCPDFactorization
from _include.m_libpgm.hybayesiannetwork import HyBayesianNetwork
from _include.m_libpgm.nodedata import NodeData

class TSCBN(HyBayesianNetwork):
    '''
    This class represents a Hybrid Temporal Dependency Bayesian network with hybrid nodes. 
    It contains the attributes *V*, *E*, *Vdata*
    and the method *randomsample*.
    '''

    def __init__(self, root, orderedskeleton=None, nodedata=None, random_gen = True, unempty= False, forbid_never=False, discrete_only = False, default_states = {}, default_is_distributed = False):
        if (orderedskeleton != None and nodedata != None):
            
            self._initialize_master_structure(orderedskeleton, nodedata)
            if random_gen:
                self.generate_random_CPDs(unempty, forbid_never)

            self.show_plot_generated = False

            # skeleton
            self.skeleton = orderedskeleton

            # initial states
            self._force_initial_state(default_states, default_is_distributed)

            # set node data now
            nd = NodeData()       
            nd.Vdata = self.Vdata
            nd.entriestoinstances()   
            self.nodes = nd.nodes
            
            # Root 
            self.root = root # to be removed
            
            # check that inputs match up
            #assert (self.root), "Root needs to be given"
            assert (sorted(self.V) == sorted(self.Vdata.keys())), ("initial_Vdata vertices did not match vertex data:", self.V, self.Vdata.keys())


    def clear_parameters(self):
        # remove all parameters
        a= 0
        for n in self.nodes:
            try:
                if isinstance(self.nodes[n].Vdataentry["cprob"], dict):
                    for k in self.nodes[n].Vdataentry["cprob"]:
                        self.nodes[n].Vdataentry["cprob"][k] = np.zeros(len(self.nodes[n].Vdataentry["cprob"][k]))
                else:
                    self.nodes[n].Vdataentry["cprob"] = np.zeros(len(self.nodes[n].Vdataentry["cprob"]))
            except:
                pass
            try:
                if isinstance(self.nodes[n].Vdataentry["hybcprob"], dict):
                    for k in self.nodes[n].Vdataentry["hybcprob"]:
                        self.nodes[n].Vdataentry["hybcprob"][k]["mean_base"] = 0.0
                        self.nodes[n].Vdataentry["hybcprob"][k]["variance"] = 0.1

                else:
                    self.nodes[n].Vdataentry["hybcprob"]["mean_base"] = 0.0
                    self.nodes[n].Vdataentry["hybcprob"]["variance"] = 0.1
                    self.nodes[n].Vdataentry["mean_base"] = 0
                    self.nodes[n].Vdataentry["variance"] =0.1

            except:
                pass

    def _force_initial_state(self, default_states, default_is_distributed):
        ''' Initial state is forced to occurr at t=0'''
        for k in default_states:
            if default_is_distributed:
                try:
                    self.Vdata[k]["cprob"][self.Vdata[k]["vals"].index("Never")] = 0.0
                except:
                    pass
                self.Vdata[k]["cprob"] = self.Vdata[k]["cprob"] / sum(self.Vdata[k]["cprob"])
            else:
                try:
                    self.Vdata[k]["cprob"] = np.zeros(len(self.Vdata[k]["cprob"]))
                    self.Vdata[k]["cprob"][self.Vdata[k]["vals"].index(default_states[k])] = 1.0
                except:
                    self.Vdata[k]["cprob"] = np.zeros(len(self.Vdata[k]["cprob"]))
                    self.Vdata[k]["cprob"][self.Vdata[k]["vals"].index(default_states[k])] = 1.0

    def update_CPDs(self):
        nd = NodeData()       
        nd.Vdata = self.Vdata
        nd.entriestoinstances()   
        self.nodes = nd.nodes
            
    # def draw(self, mode = "ext", conditions = dict()):
    #     probs = conditions
    #     if mode == "ext":
    #         V().draw_network_graph_given([f for f in self.E if not(str.startswith(f[0], "dL_") or str.startswith(f[1], "dL_"))], [f for f in self.V if not(str.startswith(f, "dL_") or str.startswith(f, "tp_"))], probs, self.Vdata, True)
    #     if mode == "int":
    #         V().draw_network_graph_given(self.E, self.V, probs, self.Vdata, True)
        
    def print_distributions(self):
        for q in self.Vdata:
            try:
                print("\n Distribution %s: %s" % (q, self.Vdata[q]["cprob"]))
            except:
                print("\n Distribution %s: %s" % (q, self.Vdata[q]["hybcprob"]))
            print("with parents: %s" % (self.Vdata[q]["parents"]))
            try:
                print("with values: %s" % (self.Vdata[q]["vals"]))
            except:
                pass
                
    def generate_random_CPDs(self, random_dists=False, forbid_never=False):

        condDict = dict()
        prob = dict()
        
        for rv_name in self.Vdata:
            #print(self.Vdata[rv_name]["type"])
            
            if self.Vdata[rv_name]["type"]  == "discrete":
                condDict[rv_name] = self._discrete_disc_par_random_CPD(rv_name, random_dists, forbid_never)
                prob[rv_name] = "cprob"
            elif self.Vdata[rv_name]["type"]  == "lg":
                condDict[rv_name] = self._continuous_cont_par_random_CPD(rv_name, random_dists)
                prob[rv_name] = None
            elif self.Vdata[rv_name]["type"]  == "lgandd":
                condDict[rv_name], len_pars = self._continuous_cont_n_disc_par_random_CPD(rv_name, random_dists)
                prob[rv_name] = "hybcprob"
                self.Vdata[rv_name]["len_pars"] = len_pars # Anzahl disc. parents
            

        ord = 1
        for k in self.Vdata:
            self.Vdata[k][prob[k]] = condDict[k]
            if not "hybcprob" is prob[k]:
                self.Vdata[k]['ord'] = ord 
            ord += 1
    
    def _discrete_disc_par_random_CPD(self, node_name, random_dists, forbid_never):
        ''' generate a CPD for the discrete case '''

        condDict2 = dict()

        # Ermittlung parents per node
        l_prob = len(self.Vdata[node_name]["vals"])
        all_vals = []
        for p in self.Vdata[node_name]["parents"]:
            all_vals.append(self.Vdata[p]["vals"])

        # Knoten bedingt auf nix
        a = np.random.rand(1,l_prob)[0]
        if forbid_never:
            try:
                n_idx = self.Vdata[node_name]["vals"].index("Never")
            except:
                pass

        if not all_vals:
            if not random_dists:
                a = np.zeros(l_prob)
                return a
            else:
                a = np.random.rand(1,l_prob)[0]
                if forbid_never:
                    try:
                        a[n_idx]=0.0
                    except:
                        pass
            return a/sum(a)




        # jeder Knoten bedingt auf seine Parents   
        if len(all_vals)>0: 
            i = 1
            LSL = [[n] for n in all_vals[0]]
            if len(all_vals)>1:                        
                while len(all_vals)!=i: 
                    nextList =all_vals[i];i+=1
                    LSLNew = []
                    for l in LSL:
                        for n in nextList:
                            LSLNew.append(l + [n])
                    LSL = LSLNew

            # set random
            #LSL += [str([])]
            for l in LSL:
                if not random_dists:
                    a = np.zeros(l_prob)
                    if forbid_never: a[n_idx] = 0.0
                    condDict2[str(l)] = a

                else:
                    a = np.random.rand(1,l_prob)[0]
                    if forbid_never:
                        try:
                            a[n_idx] = 0.0
                        except:
                            pass
                    condDict2[str(l)] = a/sum(a) # Create Random Distribution

            
        return condDict2
    
    def _continuous_cont_par_random_CPD(self, node_name, empty_dists):
        ''' generate a CPD for the discrete case 
        "mean_base": <float used for mean starting point
              (\mu_0)>,
        "mean_scal": <array of scalars by which to
                      multiply respectively ordered 
                      continuous parent outcomes>,
        "variance": <float for variance>
        
        '''
        raise NotImplementedError("Method _continuous_cont_par_random_CPD() was not required so far. If needed now -> go and implement it!")  
        return {}
       
    def _continuous_cont_n_disc_par_random_CPD(self, node_name, empty_dists):
        ''' generate a CPD for the discrete case  
            i.e. for each combination of discrete values need a continuous distribution        
        
        '''
        
        condDict2 = dict()
        
        # Ermittlung parents per node
        l_prob = 0#len(self.Vdata[node_name]["vals"])
        all_vals = []
        for p in self.Vdata[node_name]["parents"]: 
            # if discrete append
            if self.Vdata[p]["type_dc"] == "disc":
                all_vals.append(self.Vdata[p]["vals"])
        
        # Knoten bedingt auf nix
        a = np.random.rand(1,l_prob)[0]
        
        #if not empty_dists:
        #    condDict2["[]"] = a/sum(a) # Create Random Distribution
        #else:
        #    condDict2["[]"] =np.zeros(l_prob)

        # jeder Knoten bedingt auf seine Parents   
        if len(all_vals)>0: 
            i = 1
            LSL = [[n] for n in all_vals[0]]
            if len(all_vals)>1:                        
                while len(all_vals)!=i: 
                    nextList =all_vals[i];i+=1
                    LSLNew = []
                    for l in LSL:
                        for n in nextList:
                            LSLNew.append(l + [n])
                    LSL = LSLNew

            # set random
            LSL += [str([])]
            for l in LSL:                
                condDict2[str(l)] = dict()
                condDict2[str(l)]["variance"] = self.Vdata[node_name]["variance"]
                condDict2[str(l)]["mean_base"] = self.Vdata[node_name]["mean_base"]
                condDict2[str(l)]["mean_scal"] = len(all_vals)*[1]
                            
        return condDict2, len(all_vals) # in this case this dictionary is hybcprob
     
    def _initialize_master_structure(self, orderedskeleton, nodedata):
        '''
        Structure within myself - time structure - vs. value structure

        BEWUSST: Do not include edge from node to previous parent (macht keinen SINN!)
        '''
        
        # 1. Nodeinfo rausziehen
        self.Vdata = nodedata

        # 2. Force node per temporal variable to be at t = 0
        #    if for any temporal variable t != 0 add it and set to 0


        # dL abhängig immer (!) von mir und meinen Parents
        # wobei dieses dL der Abstand zu meinem Vorgänger event ist
        # 0. Create internal nodes
        for v in copy.deepcopy(orderedskeleton.V):
            dL_v = "dL_" + v
            orderedskeleton.V += ["dL_" + v]
            self.Vdata[dL_v] = dict()
            
            # only for discrete
            if  self.Vdata[v]["type_dc"] == "disc": self.Vdata[v]["numoutcomes"] = len(self.Vdata[v]["vals"])

            self.Vdata[v]["parents"] = []
            self.Vdata[v]["children"] = []
            self.Vdata[v]["type_dc"] = "disc"

            self.Vdata[dL_v]["parents"] = []
            self.Vdata[dL_v]["children"] = []
            self.Vdata[dL_v]["type_dc"] = "cont"
            self.Vdata[dL_v]["mean_base"] = self.Vdata[v]["dL_mean"]
            self.Vdata[dL_v]["variance"] = self.Vdata[v]["dL_var"]

            # my node influences the time to me
            orderedskeleton.E += [[v, dL_v]]
            self._add_par_child(v, dL_v)                        
        
        for e in copy.deepcopy(orderedskeleton.E):            
            fr = e[0]
            to = e[1]
            if str.startswith(e[0], "dL_") or str.startswith(e[0], "tp_") or str.startswith(e[1], "dL_") or str.startswith(e[1], "tp_"):
                continue
            
            # add normal value node connection - need connection 
            # to node e.g. B1 but also to preceding node e.g. B0
            self._add_par_child(fr, to)
            
            # from beeinflusst dL vom to
            orderedskeleton.E += self._add_edge_par_child(fr, "dL_" + to)

        # adjust type per node now
        for v in self.Vdata:      
            
            self.Vdata[v]["type"], self.Vdata[v]["type_parents"] = self._get_type(self.Vdata[v]["parents"], self.Vdata[v]["type_dc"])
            assert (self.Vdata[v]["type"] in ["discrete", "cont", "lg", "lgandd"]), "Found type "+str(self.Vdata[v]["type"])+" - but Type of variable has to be discrete, cont, lg or mixture of lg and discrete"
            
            
        orderedskeleton.toporder()   
        self.V = orderedskeleton.V # A list of the names of the vertices.
        self.E = orderedskeleton.E # A list of [origin, destination] pairs of vertices that make edges.
              
    def _add_edge_par_child(self, fr, to ):
        self._add_par_child(fr, to)
        return [[fr, to]]
                
    def _get_type(self, parents, my_type):
        has_disc = False
        has_cont = False
        no_parents = len(parents)==0
        specs_cont = [] # specifies which parents are cont
        specs_disc = [] # specifies which parents are disc
        for p in parents:
            if self.Vdata[p]["type_dc"] == "disc":
                specs_disc.append(p)
                has_disc=True
            if self.Vdata[p]["type_dc"] == "cont": 
                specs_cont.append(p)
                has_cont=True
        
        res = None
        if my_type == "cont": # cases: c, d, cd, none
            if (not has_disc and has_cont) or no_parents:
                res =  "lg"
            else: # disc oder disc and cont
                res =  "lgandd"
        
        if my_type == "disc":
            if (has_disc and not has_cont) or no_parents:
                res =  "discrete" # discrete and parents discrete
            if has_disc and has_cont: # TO BE IMPLEMENTED
                res =  "disParcandd"
            if not has_disc and has_cont:
                res =  "disParAllcont"

        return res, {"c" : specs_cont, "d" : specs_disc}

    def _add_par_child(self, from_node, to_node, mode="int"):
        
        if mode== "int":
            try: self.Vdata[to_node]["parents"].append(from_node)
            except: self.Vdata[to_node]["parents"] = [from_node]
            
            try: self.Vdata[from_node]["children"].append(to_node)
            except: self.Vdata[from_node]["children"] = [to_node]

        
        
