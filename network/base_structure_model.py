#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
from network.base import Base


class BaseStructureModel(Base):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        # shows the results that were generated
        self.show_plot_generated = False
        self.show_text_generated = False

        # For evaluation
        self.parameter_execution_time = 0.0


    def generate_model(self, structure_specification):
        self.not_implemented("generate_model")
        
    def model_key(self):        
        return self.__class__.__name__

    def _dynamic_node(self, node_name, type_dc, vals, node_cpds):
        node_cpds[node_name] = dict()
        node_cpds[node_name]["vals"] = vals
        node_cpds[node_name]["type_dc"] = type_dc # preliminary - will be adjusted depending on parents!
        return [node_name]

    def _self_dependencies(self, vertices, rel_nodes):
        try:
            edges = []
            verts = sorted(vertices)
            fr = verts[0]

            for v in verts[1:]:
                to = v
                fst = fr.split("_")[0] #''.join([i for i in fr if not i.isdigit()])
                if fst == to.split("_")[0] and fst in rel_nodes:
                    edges.append([fr, to])
                fr = to
            return edges
        except:
            return []