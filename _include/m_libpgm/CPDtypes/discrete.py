# Copyright (c) 2012, CyberPoint International, LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the CyberPoint International, LLC nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
This module contains tools for representing discrete nodes -- those with a finite number of outcomes and a finite number of possible parent values -- as class instances with their own *choose* method to choose an outcome for themselves based on parent outcomes.

'''
import random

class Discrete():
    '''
    This class represents a discrete node, as described above. It contains the *Vdataentry* attribute and the *choose* method.
    
    '''
    def __init__(self, Vdataentry):
        '''
        This class is constructed with the argument *Vdataentry* which must be a dict containing a dictionary entry for this particular node. The dict must contain an entry of the following form::
        
            "cprob": {
                "['<parent 1, value 1>',...,'<parent n, value 1>']": [<probability of vals[0]>, ... , <probability of vals[n-1]>],
                ...
                "['<parent 1, value j>',...,'<parent n, value k>']": [<probability of vals[0]>, ... , <probability of vals[n-1]>],
            }

        Where the keys are each possible combination of parent values and the values are the probability of each of the *n* possible node outcomes, given those parent outcomes. The *Vdataentry* attribute is set equal to this *Vdataentry* input upon instantiation.

        '''
        if not Vdataentry["parents"]:
            Vdataentry["parents"] = None

        self.Vdataentry = Vdataentry
        # set parents to None if not existing
        
        '''A dict containing CPD data for the node.'''

    def choose(self, pvalues):
        '''
        Randomly choose state of node from a probability distribution conditioned on parent values *pvalues*.

        This method has two parts: (1) determining the proper probability
        distribution, and (2) using that probability distribution to determine
        an outcome.

        Arguments:
            1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in ``self.Vdataentry["parents"]``.
        The function goes to the proper entry in *Vdataentry*, as specified by *pvalues*, and samples the node based on the distribution found there. 

        '''
        random.seed()

        p = self.Vdataentry["parents"]
        if (not p):
            distribution = self.Vdataentry["cprob"]
        else:
            #pvalues = [str(outcome[t]) for t in self.Vdataentry["parents"]] # ideally can we pull this from the skeleton so as not to store parent data at all?
            
            for pvalue in pvalues:
                assert pvalue != 'default', "Graph skeleton was not topologically ordered."
            
            distribution = self.Vdataentry["cprob"][str(pvalues)]
            
        # choose
        rand = random.random()
        lbound = 0 
        ubound = 0
        for interval in range(int(self.Vdataentry["numoutcomes"])):
            
            ubound += distribution[interval]
            if (lbound <= rand and rand < ubound):
                rindex = interval
                break
            else:
                lbound = ubound 
    
        return str(self.Vdataentry["vals"][rindex])
