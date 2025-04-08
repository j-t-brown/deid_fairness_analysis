"""
Code to run OLA global k-anonymity generalization.
"""

## Libraries
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from functools import partial
from multiprocessing import Pool


## Helper functions

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 1e-10
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def process_group_object(data, attributes, generalization_hierarchies, k, suppression_prop, name_hier,
 group_var, divergence_prior, group_value):
    
    # create standard OLA object on subset of the data
    obj = OLA(data = data[data[group_var] == group_value],
              attributes = attributes,
              generalization_hierarchies = generalization_hierarchies,
              k = k,
              suppression_prop = suppression_prop,
              name_hier = name_hier,
              group_var = group_var,
              divergence_prior = divergence_prior)

    # find k minimal generalizations
    obj.find_kmin()

    # find optimal k minimal generalizations for each utility measure
    obj.process_kmin()


    # return completed object and group value identifier
    return [group_value, obj]


## OLA class
class OLA:
    
    
    def __init__(self, data, attributes, generalization_hierarchies, k, suppression_prop, name_hier,
        group_var, util_functions = ['js_div'], divergence_prior = 1e-10):
        
        # store inputs
        self.data = data.copy()
        self.attributes = attributes.copy()
        self.hier_dict = generalization_hierarchies.copy()
        self.k = k
        self.sup_prop = suppression_prop
        self.tot_pop = data.counts.sum()
        self.name_hier = name_hier.copy()
        self.util_functions = util_functions
        self.prior = divergence_prior
        self.group_var = group_var
        self.done = False
        
        # attributes must be ordered in following manner for generalization functions
        self.ordered_attributes = ['zcta5', 'age', 'race', 'sex', 'ethnicity']
        
        # order generalization hierarchies
        self.gen_hiers = []
        for attribute in self.ordered_attributes:
            if attribute in self.attributes:
                self.gen_hiers.append(self.hier_dict[attribute])
            else:
                self.gen_hiers.append({0:False})
                self.name_hier[attribute] = {0:'*'}
        
        # initialize lattice
        self.top_node = tuple([max(hier.keys()) for hier in self.gen_hiers])
        self.bottom_node = tuple([0 for _ in self.ordered_attributes])
        self.initialize_lattices()
        
        # probabilities for divergence calculation
        self.initialize_div_probs()
        
        
    def initialize_div_probs(self):
        self.data['counts_prior'] = self.data.counts + self.prior
        self.prob_pop = self.data.counts_prior.sum()
        self.P = self.data.counts_prior / self.prob_pop
        self.group_totals = (self.data.groupby(self.group_var, sort=False)['counts_prior']
                             .transform('sum')).values
        self.group_P = np.divide(self.data.counts_prior, self.group_totals)
        
        # group total counts without prior
        self.group_counts = self.data.groupby(self.group_var).agg({'counts':'sum'})
    

    def set_group_var(self, group_var):
        self.group_var = group_var
        self.initialize_div_probs()
        
    
    def initialize_lattices(self):
        
        # full latice
        self.full_lattice = defaultdict(dict)
        possible_values = (range(self.bottom_node[i], self.top_node[i] + 1) for i in range(len(self.top_node)))
        
        for node in itertools.product(*possible_values):
            self.full_lattice[sum(node)][node] = None
            
        # k-minimal results
        self.k_minimals = {}
        
    
    def sub_lattice(self, bottom_node, top_node):
        
        lattice = defaultdict(dict)
        possible_values = (range(bottom_node[i], top_node[i] + 1) for i in range(len(top_node)))
        
        for node in itertools.product(*possible_values):
            lattice[sum(node)][node] = self.full_lattice[sum(node)][node]

        return lattice
    
    
    def height(self, lattice):
        return len(lattice.keys()) - 1
    
    
    def width(self, lattice, height):
        return len(lattice[height].keys())
    
    
    def policy_parameters(self, node):
        
        params = []
        
        for i in range(len(node)):
            params.append(self.gen_hiers[i][node[i]])
            
        return params
    
    
    def set_k(self, k):
        self.k = k
        
    
    def set_suppression_prop(self, prop):
        self.sup_prop = prop
    
    
    def is_k_anonymous(self, node):
        """
        Returns Boolean indicated whether or not proposed policy
        is k-anonymous, given defined k and suppression percentage thresholds.
        """
        
        prop_notk = self.prop_suppressed(node)
        
        if prop_notk <= self.sup_prop:
            return True
        else:
            return False
    
    
    def generalized_counts(self, geos = False, ages = False, races = False, sexes = False,
     ethnicities = False):
        """
        Generalizes census data for age, race, sex, and/or ethnicity according to a single policy.
        Returns:
        An array of the class size for each equivalence class.
        """
        temp = self.data.copy()

        groupby_terms = self.ordered_attributes.copy()
        
        # generalize geocode age, race, sex, and ethnicity values
        if geos:
            if geos != -1:
                temp['zcta5'] = temp['zcta5'].str[:geos]
        else:
            groupby_terms.remove('zcta5')

        if ages:
            if ages != -1:
                temp['age'] = pd.cut(temp['age'],ages, right=False)
        else:
            groupby_terms.remove('age')

        if races:
            if races != -1:
                for race_gen in races:
                    for key, value in race_gen.items():
                        temp.loc[temp.race.isin(value), 'race'] = key
        else:
            groupby_terms.remove('race')
        
        if not sexes:
            groupby_terms.remove('sex')

        if not ethnicities:
            groupby_terms.remove('ethnicity')
                
        self.equiv_sizes = temp.groupby(groupby_terms, sort=False, observed=False)['counts'].sum().values
        
        
    def generalized_counts_distributed(self, geos = False, ages = False, races = False, sexes = False,
     ethnicities = False):
        """
        Generalizes census data for age, race, sex, and/or ethnicity according to a single policy.
        Returns:
        An array of the equivalence class size for each original bin after generalization.
        """
        temp = self.data.copy()

        groupby_terms = self.ordered_attributes.copy()
        
        # generalize geocode age, race, sex, and ethnicity values
        if geos:
            if geos != -1:
                temp['zcta5'] = temp['zcta5'].str[:geos]
        else:
            groupby_terms.remove('zcta5')

        if ages:
            if ages != -1:
                temp['age'] = pd.cut(temp['age'],ages, right=False)
        else:
            groupby_terms.remove('age')

        if races:
            if races != -1:
                for race_gen in races:
                    for key, value in race_gen.items():
                        temp.loc[temp.race.isin(value), 'race'] = key
        else:
            groupby_terms.remove('race')
        
        if not sexes:
            groupby_terms.remove('sex')

        if not ethnicities:
            groupby_terms.remove('ethnicity')

        self.risk_counts = (temp.groupby(groupby_terms, sort=False, observed=False)
                          ['counts']
                          .transform('sum')).values
    
    
    def generalized_counts_utility(self, geos = False, ages = False, races = False, sexes = False,
     ethnicities = False):
        """
        Generalizes census data for age, race, sex, and/or ethnicity according to a single policy.
        Returns:
        An array of the redistributed counts over the org
        """
        temp = self.data.copy()

        groupby_terms = self.ordered_attributes.copy()
        
        # generalize geocode age, race, sex, and ethnicity values
        if geos:
            if geos != -1:
                temp['zcta5'] = temp['zcta5'].str[:geos]
        else:
            groupby_terms.remove('zcta5')

        if ages:
            if ages != -1:
                temp['age'] = pd.cut(temp['age'],ages, right=False)
        else:
            groupby_terms.remove('age')

        if races:
            if races != -1:
                for race_gen in races:
                    for key, value in race_gen.items():
                        temp.loc[temp.race.isin(value), 'race'] = key
        else:
            groupby_terms.remove('race')
        
        if not sexes:
            groupby_terms.remove('sex')

        if not ethnicities:
            groupby_terms.remove('ethnicity')
        
        # redistribute counts
        temp['redist'] = (temp.groupby(groupby_terms, sort=False, observed=False)
                          ['counts']
                          .transform('mean')).values
        
        # create multiplier to zero out suppressed cells
        temp['multiplier'] = (temp.groupby(groupby_terms, sort=False, observed=False)
                          ['counts']
                          .transform('sum')).values
        temp['multiplier'] = (temp['multiplier'].values >= self.k).astype(int)
        
        self.util_counts = (temp['redist'] * temp['multiplier']).values


    def generalized_counts_per_equiv(self, geos = False, ages = False, races = False, sexes = False,
     ethnicities = False):
        """
        Prepares data for self.calc_bins_per_equiv
        """
        temp = self.data.copy()

        groupby_terms = self.ordered_attributes.copy()
        
        # generalize geocode age, race, sex, and ethnicity values
        if geos:
            if geos != -1:
                temp['zcta5'] = temp['zcta5'].str[:geos]
        else:
            groupby_terms.remove('zcta5')

        if ages:
            if ages != -1:
                temp['age'] = pd.cut(temp['age'],ages, right=False)
        else:
            groupby_terms.remove('age')

        if races:
            if races != -1:
                for race_gen in races:
                    for key, value in race_gen.items():
                        temp.loc[temp.race.isin(value), 'race'] = key
        else:
            groupby_terms.remove('race')
        
        if not sexes:
            groupby_terms.remove('sex')

        if not ethnicities:
            groupby_terms.remove('ethnicity')
        
        # number of original bins per new equivalence class
        return [(temp.groupby(groupby_terms, sort=False, observed=False)['counts'].transform('count')).values,
            (temp.groupby(groupby_terms, sort=False, observed=False)['counts'].transform('sum')).values]
    
    
    def is_tagged_k_anonymous(self, node):
        if self.full_lattice[sum(node)][node] == True:
            return True
        else:
            return False
        
        
    def is_tagged_not_k_anonymous(self, node):
        if self.full_lattice[sum(node)][node] == False:
            return True
        else:
            return False
        
    
    def get_tag(self, node):
        return self.full_lattice[sum(node)][node]
    
    
    def is_tagged(self, node):
        return self.full_lattice[sum(node)][node] != None
    
    
    def tag_k_anonymous(self, node):
        self.full_lattice[sum(node)][node] = True
        self.predictive_k_anonymous(node)
        
        
    def tag_not_k_anonymous(self, node):
        self.full_lattice[sum(node)][node] = False
        self.predictive_not_k_anonymous(node)
        
        
    def predictive_not_k_anonymous(self, node):
        """
        Tags nodes that are connected to and below node as not k-anonymous.
        """
        
        lower_nodes = (range(self.bottom_node[i], node[i] + 1) for i in range(len(node)))
        
        for node in itertools.product(*lower_nodes):
            self.full_lattice[sum(node)][node] = False
            
    
    def predictive_k_anonymous(self, node):
        """
        Tags nodes that are connected to and above as k-anonymous.
        """
        
        upper_nodes = (range(node[i], self.top_node[i] + 1) for i in range(len(node)))
        
        for node in itertools.product(*upper_nodes):
            self.full_lattice[sum(node)][node] = True
            
            
    def clean_up(self, node):
        """
        Removes all nodes in self.k_minimals that are generalizations of node.
        """
        
        keep_node = True
        
        current_level = sum(node)
        
        for level, old_nodes in self.k_minimals.items():
            
            # remove more generalized nodes of current node
            if level > current_level:
                
                for old_node in old_nodes:
                    if np.all(np.array(node) <= np.array(old_node)):
                        self.k_minimals[level].remove(old_node)
            
            #if current node is more generalized than old nodes, do not add current node
            elif level <= current_level:
                
                for old_node in old_nodes:
                    if np.all(np.array(node) >= np.array(old_node)):
                        #print('Keep = False')
                        keep_node = False
                        break
                        
        if keep_node:
            if current_level in self.k_minimals.keys():
                if node not in self.k_minimals[current_level]:
                    self.k_minimals[current_level].append(node)
            else:
                self.k_minimals[current_level] = [node]
                
        #print(self.k_minimals)
        #print()
            
    
    def Kmin(self, bottom_node, top_node):

        L = self.sub_lattice(bottom_node, top_node)
        H = self.height(L)

        #print(L)
        #print()
        
        if not self.done:
            if H > 1:

                h = round(H/2)

                for node in L[list(L.keys())[h]]:
                    #print(node)

                    if self.is_tagged_k_anonymous(node):
                        self.Kmin(bottom_node, node)

                    elif self.is_tagged_not_k_anonymous(node):
                        self.Kmin(node, top_node)

                    elif self.is_k_anonymous(node):
                        self.tag_k_anonymous(node)
                        self.Kmin(bottom_node, node)

                    else:
                        self.tag_not_k_anonymous(node)
                        self.Kmin(node, top_node)

            else:

                if self.is_tagged_not_k_anonymous(bottom_node):
                    N = top_node
                elif self.is_k_anonymous(bottom_node):
                    self.tag_k_anonymous(bottom_node)
                    N = bottom_node
                    if bottom_node == self.bottom_node:
                        self.done=True
                else:
                    self.tag_not_k_anonymous(bottom_node)
                    N = top_node

                #print(N, bottom_node, top_node)
                self.clean_up(N)


    def calc_util_func(self, func, node):

        if func == 'js_div':
            return self.calc_js_div()

        if func == 'query_loss':
            return self.calc_query_loss()

        if func == 'pct_mov':
            return self.calc_abs_pct_moved()

        if func == 'nonu_ent':
            return self.calc_nonuniform_entropy()

        if func == 'dm*':
            return self.calc_DM(node)

        if func == 'num_per_equiv':
            return self.calc_bins_per_equiv(node)
    
    
    def calc_utility(self, node):

        # dataframe of results
        util_vals = pd.DataFrame(columns = self.util_functions, index=[0])
        
        # get bin sizes after generalization
        self.gen_utility_counts(node)

        for func in self.util_functions:

            util_vals[func] = self.calc_util_func(func, node)
        
        return util_vals

    
    def prop_suppressed(self, node):
        
        # get equivalence class sizes per policy parameters
        self.gen_risk_counts(node)

        # check if proportion of population not k-anonymous meets suppression percentage
        prop_notk = (self.equiv_sizes[self.equiv_sizes < self.k]).sum() / self.tot_pop

        return prop_notk
    
    
    def calc_js_div(self):
        
        # define distributions
        Q = (self.util_counts + self.prior)
        Q /= Q.sum()
        
        M = (self.P + Q) / 2
        
        return 0.5 * (sum(self.P * np.log2(self.P / M)) + 
                  sum(Q * np.log2(Q / M)))


    def calc_nonuniform_entropy(self):
        """
        Assumes that the probability of suppressed records = # records for the bin / 
        total # records in the dataset. This has the caveat that records who have their
        quasi-identifier suppressed, but not their remaining values, have the same utility
        as those records that are suppressed completely from the dataset.
        """

        # replace suppressed values with size of the dataset + 1
        util_counts = self.risk_counts.copy()
        util_counts[util_counts < self.k] = self.tot_pop + 1

        return (-np.log2(self.data.counts.values / util_counts) * self.data.counts.values).sum() / self.tot_pop


    def calc_DM(self, node):
        """
        Same assumption as for calc_nonuniform_entropy.
        Calculates DM* as defined in El Emam's OLA paper.
        """

        # replace suppressed cells with equivalence class size of tot_pop + 1
        util_counts = self.equiv_sizes.copy()
        suppressed = util_counts < self.k
        part1 = (util_counts[~suppressed] ** 2).sum()
        part2 = (util_counts[suppressed] * (self.tot_pop + 1)).sum()
        return (part1 + part2) / self.tot_pop


    def calc_bins_per_equiv(self, node):
        """
        Assumes that suppressed records are combined with all other pre-anonymization equivalence classes.
        Calculates the number of pre-anonymization equivalence classes per post-anonymization
        equivalence classes.
        """

        # get bin sizes after generalization
        bin_sizes = self.gen_counts_per_equiv(node)

        return (bin_sizes * self.data.counts.values).sum() / self.tot_pop
    
    
    def calc_query_loss(self):
        return ((np.abs(self.data.counts.values - self.util_counts) /
                 self.data.counts.values) * 100).mean()
    
    
    def calc_abs_pct_moved(self):
        return ((np.abs(self.data.counts.values - self.util_counts).sum()) /
                self.tot_pop) * 100
    
    
    def calc_overall_marketer(self):
        """
        Calculates the overall marketer risk for the given policy.
        """

        # replace suppressed cells with tot_pop
        risk_counts = self.risk_counts.copy()
        risk_counts[risk_counts < self.k] = self.tot_pop
        
        mark_risk = (np.nan_to_num(self.data.counts.values / risk_counts).sum() / self.tot_pop)
        
        return mark_risk


    def calc_initial_marketer(self):
        """
        Calculates the marketer risk of the raw dataset.
        """
        return len(self.data[self.data.counts > 0]) / self.tot_pop


    def calc_PREC(self):
        """
        Calculates Sweeney's PREC measure.
        """

        if 'prec' in self.util_functions:

            # maximum height of each generalization hierarchy
            max_height = np.array(self.top_node) + 1

            # intermediate result for not suppressed cells
            part1 = ((self.ans[self.ordered_attributes] / max_height).sum(axis=1) 
                * (1 - self.ans['prop_suppressed']))

            # intermediate result for suppressed cells
            part2 = len(self.attributes) * self.ans['prop_suppressed']

            self.ans['prec'] = 1 - ((part1 + part2)/len(self.attributes))
        
                
    def process_kmin(self):
        """
        Calculate overall utility for k minimal nodes.
        The results can identify the optimal generalization
        according to the selected utility measure.
        """

        self.ans = pd.DataFrame(columns = self.ordered_attributes)

        pk_vals = []
        marketer = []
        util_vals = []

        i = 0

        for key, nodes in self.k_minimals.items():

            for node in nodes:
                
                # store generalization levels
                self.ans.loc[i, :] = list(node)
                
                # PK value
                pk_vals.append(self.prop_suppressed(node))

                # marketer risk values
                self.gen_distributed_risk_counts(node)
                marketer.append(self.calc_overall_marketer())
                
                # utility values
                util_vals.append(self.calc_utility(node))                
                
                i += 1

        self.ans['k'] = self.k
        self.ans['suppression_prop_threshold'] = self.sup_prop
        self.ans['prop_suppressed'] = pk_vals
        self.ans['initial_marketer'] = self.calc_initial_marketer()
        self.ans['marketer'] = marketer
        self.ans['marketer_ratio'] = self.ans['marketer'] / self.ans['initial_marketer']
        self.ans[self.util_functions] = pd.concat(util_vals).values

        self.calc_PREC()    
        
        
    def add_names(self):
        """
        Add policy name column.
        """
        
        self.ans['policy'] = (self.ans[self.ordered_attributes]
                              .replace(self.name_hier)
                              .astype(str)
                              .agg(''.join, axis=1))
        
        
    def calc_group_measures(self):
        
        # initialize dataframes
        self.init_group_measures()
        
        for _, policy in self.ans.iterrows():

            # node
            node = tuple(policy[self.ordered_attributes].values)

            # k value
            self.k = int(policy['k'])
            # print(self.k)
        
            # get policy name
            self.policy_name = self.get_policy_name(node)

            # calculate marketer risk measures
            self.gen_distributed_risk_counts(node)
            self.calc_group_marketer()

            # calculate proportion suppressed
            self.calc_group_prop_suppressed()

            # calculate utility measures
            self.calc_group_utility(node)
        
        # calculate inequality
        self.calc_marketer_inequality()
        self.calc_util_inequality()


    def calc_group_utility(self, node):

        self.gen_utility_counts(node)

        for func in self.util_functions:

            self.calc_group_util_func(func, node)


    def calc_group_util_func(self, func, node):

        if func == 'js_div':
            self.calc_group_js_div()

        if func == 'query_loss':
            self.calc_group_query_loss()

        if func == 'pct_mov':
            self.calc_group_abs_pct_moved()

        if func == 'nonu_ent':
            self.calc_group_nonuniform_entropy()

        if func == 'dm*':
            self.calc_group_DM()

        if func == 'num_per_equiv':
            self.calc_group_bins_per_equiv(node)

        if func == 'prec':
            self.calc_group_PREC(node)
        

    def get_policy_name(self, node):
        
        name = ''
        for i in range(len(node)):
            
            attr = self.ordered_attributes[i]
            attr_level = node[i]
            
            if attr in self.name_hier.keys():
                name += self.name_hier[attr][attr_level]
            else:
                name += '*'
                
        return name
        
        
    def init_group_measures(self):
        
        group_vars = list(self.data[self.group_var].squeeze().unique())
        
        self.group_marketer = pd.DataFrame(index = group_vars)
        self.group_js = pd.DataFrame(index = group_vars)
        self.group_query_loss = pd.DataFrame(index = group_vars)
        self.group_pct_moved = pd.DataFrame(index = group_vars)
        self.group_prop_suppr = pd.DataFrame(index = group_vars)
        self.group_nonu_ent = pd.DataFrame(index = group_vars)
        self.group_dm_star = pd.DataFrame(index = group_vars)
        self.group_num_per_equiv = pd.DataFrame(index = group_vars)
        self.group_prec = pd.DataFrame(index = group_vars)
        
        
    def calc_group_marketer(self):

        # replace suppressed cells with tot_pop
        risk_counts = self.risk_counts.copy()
        risk_counts[risk_counts < self.k] = self.tot_pop
        
        # marketer risk fractions
        mark_df = (pd.concat([self.data[self.group_var],
            pd.DataFrame(data = np.nan_to_num(self.data.counts.values /risk_counts))],
            axis = 1)
        .reset_index(drop=True))
        
        # sum fractions by group
        group_mark_fracs = mark_df.groupby(self.group_var).sum()
        
        # save group-specific marketer risk
        self.group_marketer = (self.group_marketer.merge(
                                           pd.DataFrame({self.policy_name : 
                                                         (np.nan_to_num(group_mark_fracs.values /
                                                                        self.group_counts.values)
                                                            .ravel())},
                                                        index = self.group_counts.index),
                                           left_index=True,
                                           right_index=True,
                                           how='left')).fillna(0)


    def gen_counts_per_equiv(self, node):

        # get policy parameters from generalization hierarchies
        pol_params = self.policy_parameters(node)

        # get bin sizes after generalization
        bin_sizes, risk_counts = self.generalized_counts_per_equiv(geos = pol_params[0],
                                        ages = pol_params[1],
                                        races = pol_params[2],
                                        sexes = pol_params[3],
                                        ethnicities = pol_params[4])

        # change suppressed cells number of equivalence classes to number classes suppres
        bin_sizes[risk_counts < self.k] = sum(risk_counts < self.k)
        return bin_sizes
        
        
    def gen_utility_counts(self, node):
        
        # get policy parameters from generalization hierarchies
        pol_params = self.policy_parameters(node)

        # get bin sizes after generalization
        self.generalized_counts_utility(geos = pol_params[0],
                                        ages = pol_params[1],
                                        races = pol_params[2],
                                        sexes = pol_params[3],
                                        ethnicities = pol_params[4])
        
        
    def gen_risk_counts(self, node):
        
        # get policy parameters from generalization hierarchies
        pol_params = self.policy_parameters(node)

        # get bin sizes after generalization
        self.generalized_counts(geos = pol_params[0],
                                ages = pol_params[1],
                                races = pol_params[2],
                                sexes = pol_params[3],
                                ethnicities = pol_params[4])
        
    
    def gen_distributed_risk_counts(self, node):
        
        # get policy parameters from generalization hierarchies
        pol_params = self.policy_parameters(node)

        # get bin sizes after generalization
        self.generalized_counts_distributed(geos = pol_params[0],
                                            ages = pol_params[1],
                                            races = pol_params[2],
                                            sexes = pol_params[3],
                                            ethnicities = pol_params[4])
        

    def calc_group_js_div(self):
        """
        Group-specific JS divergence.
        """
        
        # define group_specific probability distributions
        Q_totals = (pd.concat([self.data[self.group_var],
                               pd.DataFrame(data = (self.util_counts + self.prior))],
                               axis = 1)
                      .reset_index(drop=True)
                      .groupby(self.group_var, sort=False)
                      .transform('sum')).values.squeeze()
        
        Q = np.divide((self.util_counts + self.prior), Q_totals)
        M = (self.group_P + Q) / 2
        
        # JS divergence
        js_div = 0.5 * ((self.group_P * np.log2(self.group_P / M)) + 
                  (Q * np.log2(Q / M)))
        
        # put in dataframe
        js_df = pd.DataFrame({self.policy_name : js_div})
        
        # add group columns
        js_df[self.group_var] = self.data[self.group_var]
        
        # divergence values by group
        self.group_js = self.group_js.merge(js_df.groupby(self.group_var).agg({self.policy_name:'sum'}),
                                            left_index = True,
                                            right_index = True,
                                            how = 'left')
        
    
    def calc_group_query_loss(self):
        """
        Mimics the execution of all COUNT queries for each quasi-identifier combination
        at the most specific values. Evalutes the average percent difference across all queries,
        broken down by pre-specified group.
        """

        # construct dataframe
        u_df = self.data[self.group_var + ['counts']].copy()
        u_df['util_counts'] = self.util_counts
        u_df['abs_pct_diff'] = (np.abs(u_df.counts.values - u_df.util_counts.values) / u_df.counts.values) * 100

        # utility loss values by group
        group_loss = u_df.groupby(self.group_var).agg({'abs_pct_diff':'mean'})
        group_loss.columns = [self.policy_name]

        self.group_query_loss = self.group_query_loss.merge(group_loss,
                                                            left_index=True,
                                                            right_index=True,
                                                            how='left')
        
    
    def calc_group_abs_pct_moved(self):
        """
        Mimics the execution of all COUNT queries for each quasi-identifier combination
        at the most specific values. Calculates the cumulative absolute difference across
        all queries, and then divides by the size of the population. Does this for each
        pre-specified group.

        Essentially indicates the percent of the population into which anonymization has induced enough
        uncertainty to "move" them to another bin.
        """

        # construct dataframe
        u_df = self.data[self.group_var + ['counts']].copy()
        u_df['util_counts'] = self.util_counts
        u_df['abs_diff'] = (np.abs(u_df.counts.values - u_df.util_counts.values) / u_df.counts.values)

        # calculate group specific values
        grouped = u_df.groupby(self.group_var).agg(cum_diff = ('abs_diff', 'sum'),
                                                   total_pop = ('counts', 'sum'))

        grouped[self.policy_name] = (grouped.cum_diff.values / grouped.total_pop.values) * 100

        # utility loss values by group
        self.group_pct_moved = self.group_pct_moved.merge(grouped[self.policy_name],
                                                          left_index=True,
                                                          right_index=True,
                                                          how='left')
        
        
    def calc_group_prop_suppressed(self):
        """
        Calculates percent of group whose records are suppressed under the policy.
        """
        
        # calculate
        df = (self.data[self.group_var + ['counts']]).copy()
        df[self.policy_name] = df.counts * (self.risk_counts < self.k) # mark suppressed
        group_suppr = ((df.groupby(self.group_var).agg({self.policy_name:'sum'})) /
                       self.group_counts.values)
        
        # store results
        self.group_prop_suppr = self.group_prop_suppr.merge(group_suppr,
                                                          left_index=True,
                                                          right_index=True,
                                                          how='left')


    def calc_group_nonuniform_entropy(self):
        """
        Calculates group-specific non-uniform entropy.
        """

        # replace zero values with size of the dataset + 1
        util_counts = self.risk_counts.copy()
        util_counts[util_counts  < self.k] = self.tot_pop + 1

        # sum and groupby group variable
        df = self.data[self.group_var + ['counts']].copy()
        df['entropy'] = -np.log2(df.counts.values / util_counts) * df.counts.values
        grouped = df.groupby(self.group_var).agg({'counts':'sum', 'entropy':'sum'})
        grouped[self.policy_name] = grouped['entropy'] / grouped['counts']

        # store results
        self.group_nonu_ent = self.group_nonu_ent.merge(grouped[[self.policy_name]],
                                                          left_index=True,
                                                          right_index=True,
                                                          how='left')


    def calc_group_PREC(self, node):
        """
        Calculates group-specific PREC.
        """

        # policy heights
        node = np.array(node)

        # maximum height of each generalization hierarchy
        max_height = np.array(self.top_node) + 1

        # group prop suppressed for policy/node
        prop_suppr = self.group_prop_suppr[self.policy_name]

        # intermediate result for not suppressed cells
        part1 = ((node / max_height).sum() * (1 - prop_suppr))

        # intermediate result for suppressed cells
        part2 = len(self.attributes) * prop_suppr

        # final prec result
        group_prec = (1 - (part1 + part2)/len(self.attributes))

        # store results
        self.group_prec = self.group_prec.merge(group_prec,
                                                left_index=True,
                                                right_index=True,
                                                how='left')


    def calc_group_bins_per_equiv(self, node):
        """
        Assumes that suppressed records are combined with all other pre-anonymization equivalence classes.
        Calculates the number of pre-anonymization equivalence classes per post-anonymization
        equivalence classes.
        """

        # get bin sizes after generalization
        bin_sizes = self.gen_counts_per_equiv(node)

        # sum by group_var
        df = self.data[self.group_var + ['counts']].copy()
        df['group_bins'] = (bin_sizes * df.counts.values)
        grouped = df.groupby(self.group_var).agg({'counts':'sum', 'group_bins':'sum'})
        grouped[self.policy_name] = grouped['group_bins'] / grouped['counts']

        # store results
        self.group_num_per_equiv = self.group_num_per_equiv.merge(grouped[[self.policy_name]],
                                                left_index=True,
                                                right_index=True,
                                                how='left')


    def calc_group_DM(self):
        """
        Groups-specific DM*
        """

        # replace suppressed values with size of the dataset + 1
        util_counts = self.risk_counts.copy()
        util_counts[util_counts  < self.k] = self.tot_pop + 1

        # sum and groupby group variable
        df = self.data[self.group_var + ['counts']].copy()
        df['equiv_class_mult'] = util_counts * df.counts.values
        grouped = df.groupby(self.group_var).agg({'counts':'sum', 'equiv_class_mult':'sum'})
        grouped[self.policy_name] = grouped['equiv_class_mult'] / grouped['counts']

        # store results
        self.group_dm_star = self.group_dm_star.merge(grouped[[self.policy_name]],
                                                left_index=True,
                                                right_index=True,
                                                how='left')
        

    def calc_marketer_inequality(self):

        gini_coeffs = np.apply_along_axis(gini, 0, self.group_marketer.values)

        self.ans['marketer_inequality'] = gini_coeffs


    def calc_util_inequality_func(self, func):

        if func == 'js_div':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_js.values)
            self.ans['js_div_inequality'] = gini_coeffs

        if func == 'query_loss':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_query_loss.values)
            self.ans['query_loss_inequality'] = gini_coeffs

        if func == 'pct_mov':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_pct_moved.values)
            self.ans['pct_mov_inequality'] = gini_coeffs

        if func == 'nonu_ent':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_nonu_ent.values)
            self.ans['nonu_ent_inequality'] = gini_coeffs

        if func == 'dm*':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_dm_star.values)
            self.ans['dm*_inequality'] = gini_coeffs

        if func == 'num_per_equiv':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_num_per_equiv.values)
            self.ans['num_per_equiv_inequality'] = gini_coeffs

        if func == 'prec':
            gini_coeffs = np.apply_along_axis(gini, 0, self.group_prec.values)
            self.ans['prec_inequality'] = gini_coeffs
        
    
    def calc_util_inequality(self):

        # utility functions
        for func in self.util_functions:

            self.calc_util_inequality_func(func)
        
        # prop suppressed
        gini_coeffs_ps = np.apply_along_axis(gini, 0, self.group_prop_suppr.values)
        self.ans['prop_suppr_inequality'] = gini_coeffs_ps


    def find_kmin(self):

        self.Kmin(self.bottom_node, self.top_node)
            





## fairU classes
class OLA_fair_sub(OLA):


    def __init__(self, data, attributes, generalization_hierarchies, target_k, min_k, 
                 suppression_prop, tot_pop, name_hier, group_label, util_function,
                 util_func_dir, divergence_prior = 1e-10):
        
        # store inputs
        self.data = data.copy()
        self.attributes = attributes.copy()
        self.hier_dict = generalization_hierarchies.copy()
        self.k = target_k
        self.target_k = target_k
        self.min_k = min_k
        self.sup_prop = suppression_prop
        self.tot_pop = tot_pop
        self.group_pop = self.data.counts.sum()
        self.name_hier = name_hier.copy()
        self.util_functions = [util_function] # one of: ['js_div', 'pct_mov', 'query_loss', 'nonu_ent', 'dm*', 'num_per_equiv', 'prec']
        self.util_func_dir = util_func_dir
        self.prior = divergence_prior
        self.group_label = group_label
        self.done = False
        
        # attributes must be ordered in following manner for generalization functions
        self.ordered_attributes = ['zcta5', 'age', 'race', 'sex', 'ethnicity']
        
        # order generalization hierarchies
        self.gen_hiers = []
        for attribute in self.ordered_attributes:
            if attribute in self.attributes:
                self.gen_hiers.append(self.hier_dict[attribute])
            else:
                self.gen_hiers.append({0:False})
                self.name_hier[attribute] = {0:'*'}
        
        # initialize lattice
        self.top_node = tuple([max(hier.keys()) for hier in self.gen_hiers])
        self.bottom_node = tuple([0 for _ in self.ordered_attributes])
        self.initialize_lattices()
        
        # probabilities for divergence calculation
        if self.util_functions[0] == 'js_div':
            self.initialize_div_probs()


    def initialize_div_probs(self):
        self.data['counts_prior'] = self.data.counts + self.prior
        self.prob_pop = self.data.counts_prior.sum()
        self.P = self.data.counts_prior / self.prob_pop


    def calc_nonuniform_entropy(self):
        """
        Assumes that the probability of suppressed records = # records for the bin / 
        total # records in the dataset. This has the caveat that records who have their
        quasi-identifier suppressed, but not their remaining values, have the same utility
        as those records that are suppressed completely from the dataset.
        """

        # replace suppressed values with size of the dataset + 1
        util_counts = self.risk_counts.copy()
        util_counts[util_counts < self.k] = self.tot_pop + 1

        return (-np.log2(self.data.counts.values / util_counts) * self.data.counts.values).sum() / self.group_pop


    def calc_DM(self, node):
        """
        Same assumption as for calc_nonuniform_entropy.
        Calculates DM* as defined in El Emam's OLA paper.
        """

        # replace suppressed cells with equivalence class size of tot_pop + 1
        util_counts = self.equiv_sizes.copy()
        suppressed = util_counts < self.k
        part1 = (util_counts[~suppressed] ** 2).sum()
        part2 = (util_counts[suppressed] * (self.tot_pop + 1)).sum()
        return (part1 + part2) / self.group_pop


    def calc_bins_per_equiv(self, node):
        """
        Assumes that suppressed records are combined with all other pre-anonymization equivalence classes.
        Calculates the number of pre-anonymization equivalence classes per post-anonymization
        equivalence classes.
        """

        # get bin sizes after generalization
        bin_sizes = self.gen_counts_per_equiv(node)

        return (bin_sizes * self.data.counts.values).sum() / self.group_pop
    
    
    def calc_query_loss(self):
        return ((np.abs(self.data.counts.values - self.util_counts) /
                 self.data.counts.values) * 100).mean()
    
    
    def calc_abs_pct_moved(self):
        return ((np.abs(self.data.counts.values - self.util_counts).sum()) /
                self.group_pop) * 100


    def prop_suppressed(self, node):
        
        # get equivalence class sizes per policy parameters
        self.gen_risk_counts(node)

        # check if proportion of population not k-anonymous meets suppression percentage
        prop_notk = (self.equiv_sizes[self.equiv_sizes < self.k]).sum() / self.group_pop

        return prop_notk


    def calc_overall_marketer(self):
        """
        Calculates the overall marketer risk for the given policy.
        """

        # replace suppressed cells with tot_pop
        risk_counts = self.risk_counts.copy()
        risk_counts[risk_counts < self.k] = self.tot_pop
        
        mark_risk = (np.nan_to_num(self.data.counts.values / risk_counts).sum() / self.group_pop)
        
        return mark_risk


    def calc_initial_marketer(self):
        """
        Calculates the marketer risk of the raw dataset.
        """
        return len(self.data[self.data.counts > 0]) / self.group_pop


    def process_kmin_sub(self):
        """
        Calculate overall utility for k minimal nodes.
        The results can identify the optimal generalization
        according to the selected utility measure.
        """

        self.ans = pd.DataFrame(columns = self.ordered_attributes)

        pk_vals = []
        util_vals = []
        marketer_vals = []

        i = 0

        for key, nodes in self.k_minimals.items():

            for node in nodes:
                
                # store generalization levels
                self.ans.loc[i, :] = list(node)
                
                # PK value
                pk_vals.append(self.prop_suppressed(node))

                # marketer risk values
                self.gen_distributed_risk_counts(node)
                marketer_vals.append(self.calc_overall_marketer())
                
                # utility values
                util_vals.append(self.calc_utility(node))
                
                i += 1

        self.ans['prop_suppressed'] = pk_vals
        self.ans['k'] = self.k
        self.ans['suppression_prop_threshold'] = self.sup_prop
        self.ans['group'] = self.group_label
        self.ans['group_pop'] = self.group_pop
        self.ans[self.util_functions] = pd.concat(util_vals).values
        self.ans['marketer'] = marketer_vals

        self.calc_PREC()


    def get_best_ans(self):

        if self.util_func_dir == 1:
            self.best_ans = self.ans.sort_values(self.util_functions, ascending=False).iloc[0].copy()
        elif self.util_func_dir == -1:
            self.best_ans = self.ans.sort_values(self.util_functions, ascending=True).iloc[0].copy()


    def meet_utility_threshold(self):

        if self.util_func_dir == 1:

            if self.best_ans[self.util_functions[0]] >= self.util_threshold:
                return True

            else:
                return False

        elif self.util_func_dir == -1:

            if self.best_ans[self.util_functions[0]] <= self.util_threshold:
                return True

            else:
                return False


    def set_utility_threshold(self, threshold):
        self.util_threshold = threshold


    def full_process_at_k(self, k):

        if k == 1:

            self.set_k(k)
            self.k_minimals = {}
            self.k_minimals[0] = {self.bottom_node}
            self.process_kmin_sub()
            self.get_best_ans()

        else:

            self.set_k(k)
            self.initialize_lattices()
            self.done = False
            self.find_kmin()
            self.process_kmin_sub()        
            self.get_best_ans()

    def k_search(self):

        """
        Find maximum k that meets utility threshold
        """

        self.get_best_ans() # <- is this needed?

        if self.meet_utility_threshold():
            pass #self.full_process_at_k(self.target_k)

        else:
            new_k = self.binary_search(low=self.min_k, high=self.target_k)

            self.full_process_at_k(new_k)


    def binary_search(self, low, high):

        """
        Binary search for maximum k value that meets
        utility threshold.
        """

        # breaking criteria
        if (high - low) == 1:

            # full process for high
            self.full_process_at_k(high)

            # if it meets utility threshold
            if self.meet_utility_threshold():
                return high
            # then the lower value must meet utility threshold
            # or is equal to self.min_k, which is as low a k value
            # as we can go.    
            else:
                self.full_process_at_k(low)
                return low

        # recursion criteria
        else:
            mid = (high + low) // 2
            self.full_process_at_k(mid)

            if self.meet_utility_threshold():
                return self.binary_search(mid, high)

            else:
                return self.binary_search(low, mid)


    def process_final_ans(self):

        self.add_names()
        self.get_best_ans()



def initialize_fairU(data, attributes, generalization_hierarchies, target_k, min_k,
 suppression_prop, tot_pop, name_hier, group_var, util_function, util_func_dir, group_label):
    
    # create OLA_fair_sub object
    obj = OLA_fair_sub(data = data[data[group_var] == group_label],
        attributes = attributes,
        generalization_hierarchies = generalization_hierarchies,
        target_k = target_k,
        min_k = min_k,
        suppression_prop = suppression_prop,
        tot_pop = tot_pop,
        name_hier = name_hier,
        group_label = group_label,
        util_function = util_function,
        util_func_dir = util_func_dir)

    # process at target k
    obj.full_process_at_k(k = target_k)

    return obj


def binary_search_helper(obj):

    obj.k_search()

    print(obj.group_label, ':', obj.k, ',', obj.best_ans[obj.util_functions[0]])

    return obj



class OLA_fairU:


    def __init__(self, data, attributes, generalization_hierarchies, target_k, 
                 suppression_prop, name_hier, group_var, delta_u, n_cores, min_k=1,
                 util_function = 'js_div', divergence_prior = 1e-10):
        
        # store inputs
        self.data = data.copy()
        self.attributes = attributes.copy()
        self.hier_dict = generalization_hierarchies.copy()
        self.target_k = target_k
        self.min_k = min_k
        self.sup_prop = suppression_prop
        self.tot_pop = data.counts.sum()
        self.name_hier = name_hier.copy()
        self.delta_u = delta_u
        self.n_cores = n_cores
        self.util_function = util_function # one of: ['js_div', 'pct_mov', 'query_loss', 'nonu_ent', 'dm*', 'num_per_equiv', 'prec']
        self.prior = divergence_prior
        self.group_var = group_var # subgroups defined by this variable - group-specific measures also based on this variable
        self.done = False
        self.group_values = list(self.data[self.group_var].squeeze().unique())
        
        # attributes must be ordered in following manner for generalization functions
        self.ordered_attributes = ['zcta5', 'age', 'race', 'sex', 'ethnicity']

        # remove subgroup defining variable
        if group_var in self.attributes:
            self.attributes.remove(self.group_var)
        
        # order generalization hierarchies
        self.gen_hiers = []
        for attribute in self.ordered_attributes:
            if attribute in self.attributes:
                self.gen_hiers.append(self.hier_dict[attribute])
            else:
                self.gen_hiers.append({0:False})

                if (attribute in self.group_var) or (attribute == self.group_var):
                    self.name_hier[attribute] = {0:'|full-{}|'.format(self.group_var)}
                else:
                    self.name_hier[attribute] = {0:'*'}

        # define whether to minimize or maximize utility function
        if util_function in ['js_div', 'pct_mov', 'query_loss', 'nonu_ent', 'dm*', 'num_per_equiv']:
            self.util_func_dir = -1
        elif util_function in ['prec']:
            self.util_func_dir = 1


    def initialize_at_target(self):

        print('Initializing at target k ---------')

        # initialize pool object
        pool = Pool(processes = self.n_cores)

        # execute
        results = pool.map_async(partial(initialize_fairU,
                                         self.data,
                                         self.attributes,
                                         self.hier_dict,
                                         self.target_k,
                                         self.min_k,
                                         self.sup_prop,
                                         self.tot_pop,
                                         self.name_hier,
                                         self.group_var,
                                         self.util_function,
                                         self.util_func_dir),
                                 self.group_values).get()

        # close object
        pool.close()
        pool.join()


        # store results
        self.OLA_objs = {}

        for result in results:

            self.OLA_objs[result.group_label] = result


    def find_utility_threshold(self):

        if self.util_func_dir == 1:
            max_utility = 0
        elif self.util_func_dir == -1:
            max_utility = np.inf
        max_group = ''

        print('Finding maximum utility ---------')

        # find maximum utility
        for group, obj in self.OLA_objs.items():

            group_util = obj.best_ans[self.util_function]

            print(group, ':', group_util)

            if self.util_func_dir == 1:

                if group_util > max_utility:

                    max_utility = group_util
                    max_group = group

            elif self.util_func_dir == -1:

                if group_util < max_utility:

                    max_utility = group_util
                    max_group = group

        # define utility threshold
        if self.util_func_dir == 1:

            self.utility_threshold = max_utility * (1 - self.delta_u)

        elif self.util_func_dir == -1:

            self.utility_threshold = max_utility * (1 + self.delta_u)

        print()
        print('Maximum utility:', max_utility)
        print('Group with max:', max_group)
        print('Utility threshold:', self.utility_threshold)


    def set_utility_threshold(self):

        print('Setting divergence threshold ---------')

        for group, obj in self.OLA_objs.items():

            obj.set_utility_threshold(self.utility_threshold)


    def binary_search_for_k(self):

        print('Binary search for new k ---------')

        # initialize pool object
        pool = Pool(processes = self.n_cores)

        # execute
        results = pool.map_async(binary_search_helper, self.OLA_objs.values()).get()

        # close object
        pool.close()
        pool.join()


        # store results
        self.OLA_objs = {}

        for result in results:

            self.OLA_objs[result.group_label] = result


    def get_all_counts(self):

        all_base_counts = []
        all_risk_counts = []
        all_util_counts = []

        # iterate through groups
        for group, obj in self.OLA_objs.items():

            # node corresponding to group's optimal generalization
            node = tuple(obj.best_ans[obj.ordered_attributes].values)

            # base counts
            all_base_counts.append(obj.data.counts.values.copy())

            # risk counts
            obj.gen_distributed_risk_counts(node)
            all_risk_counts.append(self.process_risk_counts(obj.k, obj.risk_counts))


            # utility counts
            if self.util_function in ['js_div', 'pct_mov', 'query_loss']:

                obj.gen_utility_counts(node)
                all_util_counts.append(obj.util_counts.copy())

            elif self.util_function in ['num_per_equiv']:

                all_util_counts.append(obj.gen_counts_per_equiv(node))

            # elif self.util_function in ['nonu_ent']:    <- covered with risk counts above

            #     obj.gen_distributed_risk_counts(node)
            #     all_util_counts.append(obj.risk_counts.copy())

            # elif self.util_function in ['dm*']:  <- can be calculated from subgroups' DM* values

            #     obj.gen_risk_counts(node)
            #     all_util_counts.append(obj.equiv_sizes.copy())

        self.base_counts = np.concatenate(all_base_counts)
        self.risk_counts = np.concatenate(all_risk_counts)
        if len(all_util_counts) > 0:
            self.util_counts = np.concatenate(all_util_counts)


    def process_risk_counts(self, k, counts):
        """
        Processes risk counts according to subgroup's final k value for downstream calculations.
        """

        counts = counts.copy()
        counts[counts < k] = 0
        return counts


    def calc_js_div(self):
        
        # define distributions
        P = (self.base_counts + self.prior)
        P /= P.sum()
        Q = (self.util_counts + self.prior)
        Q /= Q.sum()
        
        M = (P + Q) / 2
        
        return 0.5 * (sum(P * np.log2(P / M)) + 
                  sum(Q * np.log2(Q / M)))


    def calc_abs_pct_moved(self):
        return ((np.abs(self.base_counts - self.util_counts).sum()) /
                self.tot_pop) * 100


    def calc_query_loss(self):
        return ((np.abs(self.base_counts - self.util_counts) /
                 self.data.counts.values) * 100).mean()


    def calc_overall_marketer(self):
        """
        Calculates the overall marketer risk for the given policy.
        """

        # replace suppressed cells with tot_pop
        risk_counts = self.risk_counts.copy()
        risk_counts[risk_counts == 0] = self.tot_pop
        
        mark_risk = (np.nan_to_num(self.base_counts / risk_counts).sum() / self.tot_pop)
        
        return mark_risk


    def calc_nonuniform_entropy(self):
        """
        Assumes that the probability of suppressed records = # records for the bin / 
        total # records in the dataset. This has the caveat that records who have their
        quasi-identifier suppressed, but not their remaining values, have the same utility
        as those records that are suppressed completely from the dataset.
        """

        # replace zero values with size of the dataset + 1
        util_counts = self.risk_counts.copy()
        util_counts[util_counts == 0] = self.tot_pop + 1

        return (-np.log2(self.base_counts / util_counts) * self.base_counts).sum() / self.tot_pop


    def calc_DM(self):
        """
        Same assumption as for calc_nonuniform_entropy.
        Calculates DM* as defined in El Emam's OLA paper.
        """
        total = 0

        # iterate through items
        for group, obj in self.OLA_objs.items():

            total += (obj.best_ans[self.util_function] * obj.best_ans['group_pop'])

        return total / self.tot_pop


    def calc_bins_per_equiv(self):
        """
        Assumes that suppressed records are combined with all other pre-anonymization equivalence classes.
        Calculates the number of pre-anonymization equivalence classes per post-anonymization
        equivalence classes.
        """

        return (self.base_counts * self.util_counts).sum() / self.tot_pop


    def calc_PREC(self):
        """
        Calculates Sweeney's PREC measure. <- FIX
        """

        self.ans['overall_prec'] = (self.ans['prec'] * self.ans['group_pop']).sum() / self.tot_pop


    def calc_overall_utility(self):

        self.get_all_counts()

        func = self.util_function

        if func == 'js_div':
            self.ans['overall_' + func] = self.calc_js_div()

        if func == 'pct_mov':
            self.ans['overall_' + func] = self.calc_abs_pct_moved()

        if func == 'query_loss':
            self.ans['overall_' + func] = self.calc_query_loss()

        if func == 'nonu_ent':
            self.ans['overall_' + func] = self.calc_nonuniform_entropy()

        if func == 'dm*':
            self.ans['overall_' + func] = self.calc_DM()

        if func == 'num_per_equiv':
            self.ans['overall_' + func] = self.calc_bins_per_equiv()

        if func == 'pct_mov':
            self.ans['overall_' + func] = self.calc_PREC()


    def calc_overall_prop_suppressed(self):

        self.ans['overall_prop_suppressed'] = (self.ans['prop_suppressed'] * self.ans['group_pop']).sum() / self.tot_pop


    def process_output(self):

        # combine group details
        ans = []

        for group, obj in self.OLA_objs.items():

            ans.append(obj.best_ans)

        self.ans = pd.DataFrame(ans).reset_index(drop=True)

        # calculate overall utility
        self.calc_overall_utility()

        # calculate overall proportion suppressed
        self.calc_overall_prop_suppressed()

        # calculate overall marketer risk
        self.ans['overall_marketer'] = self.calc_overall_marketer()

        # add parameter values
        self.ans['target_k'] = self.target_k
        self.ans['min_k'] = self.min_k
        self.ans['delta_u'] = self.delta_u
        self.ans['utility_threshold'] = self.utility_threshold


    def run(self):
        self.initialize_at_target()
        self.find_utility_threshold()
        self.set_utility_threshold()
        self.binary_search_for_k()
        self.process_output()


## Subgroup-specific OLA implementation. 
## Currently, groups are designated by values for a single attribute - e.g., race.
## Needs updating to be in line with OLA and OLA_fairU classes.
# class OLA_group_specific:


#     def __init__(self, data, attributes, generalization_hierarchies, k, suppression_prop, name_hier,
#                 group_var, divergence_prior = 1e-10):
        
#         # store inputs
#         self.data = data.copy()
#         self.attributes = attributes.copy()
#         self.hier_dict = generalization_hierarchies.copy()
#         self.k = k
#         self.sup_prop = suppression_prop
#         self.tot_pop = data.counts.sum()
#         self.name_hier = name_hier.copy()
#         self.prior = divergence_prior
#         self.group_var = group_var # subgroups defined by this variable - group-specific measures also based on this variable
#         self.done = False
#         self.group_values = list(self.data[self.group_var].squeeze().unique())
        
#         # attributes must be ordered in following manner for generalization functions
#         self.ordered_attributes = ['zcta5', 'age', 'race', 'sex', 'ethnicity']

#         # remove subgroup defining variable
#         if group_var in self.attributes:
#             self.attributes.remove(self.group_var)
        
#         # order generalization hierarchies
#         self.gen_hiers = []
#         for attribute in self.ordered_attributes:
#             if attribute in self.attributes:
#                 self.gen_hiers.append(self.hier_dict[attribute])
#             else:
#                 self.gen_hiers.append({0:False})

#                 if (attribute in self.group_var) or (attribute == self.group_var):
#                     self.name_hier[attribute] = {0:'|full-{}|'.format(self.group_var)}
#                 else:
#                     self.name_hier[attribute] = {0:'*'}


#     def add_names_to_frame(self, df):
#         """
#         Add policy name column.
#         """
        
#         df['policy'] = (df[self.ordered_attributes]
#                           .replace(self.name_hier)
#                           .astype(str)
#                           .agg(''.join, axis=1))


#     def find_all_kmin(self, n_cores):

#         # initialize pool object
#         pool = Pool(processes = n_cores)


#         # execute
#         results = pool.map_async(partial(process_group_object,
#                                          self.data,
#                                          self.attributes,
#                                          self.hier_dict,
#                                          self.k,
#                                          self.sup_prop,
#                                          self.name_hier,
#                                          self.group_var,
#                                          self.prior),
#                                  self.group_values).get()

#         # close object
#         pool.close()
#         pool.join()

#         # store results in separate lists
#         self.group_values_ordered = []
#         self.kmin_objects = []

#         # concatenate results
#         for result in results:

#             self.group_values_ordered.append(result[0])
#             self.kmin_objects.append(result[1])


#     def get_all_policy_details(self, util_measure):

#         policy_details = []

#         for kmin in self.kmin_objects:
            
#             best_policy = kmin.ans.sort_values(util_measure).iloc[0]
#             policy_details.append(best_policy)
            
#         self.all_policy_details = pd.concat(policy_details, axis=1).T.reset_index(drop=True)
#         self.all_policy_details['group'] = self.group_values_ordered
#         self.all_policy_details['util_measure'] = util_measure


#     def get_group_measures(self):

#         self.group_marketer = self.all_policy_details[['marketer', 'group']].set_index('group')
#         self.group_js = self.all_policy_details[['js_div', 'group']].set_index('group')
#         self.group_query_loss = self.all_policy_details[['query_loss', 'group']].set_index('group')
#         self.group_pct_moved = self.all_policy_details[['abs_pct_moved', 'group']].set_index('group')
#         self.group_prop_suppr = self.all_policy_details[['prop_suppressed', 'group']].set_index('group')


#     def get_all_counts(self):

#         all_base_counts = []
#         all_util_counts = []
#         all_risk_counts = []

#         # iterate through groups
#         for i in range(len(self.kmin_objects)):

#             # group OLA object
#             obj = self.kmin_objects[i]

#             # node corresponding to group's optimal generalization
#             node = tuple(self.all_policy_details.iloc[i][obj.ordered_attributes].values)

#             # base counts
#             all_base_counts.append(obj.data.counts.values.copy())

#             # risk counts
#             obj.gen_distributed_risk_counts(node)
#             all_risk_counts.append(obj.risk_counts.copy())

#             # utility counts
#             obj.gen_utility_counts(node)
#             all_util_counts.append(obj.util_counts.copy())

#         self.base_counts = np.concatenate(all_base_counts)
#         self.risk_counts = np.concatenate(all_risk_counts)
#         self.util_counts = np.concatenate(all_util_counts)


#     def calc_all_group_js_div(self):

#         # define distributions
#         P = (self.base_counts + self.prior)
#         P /= P.sum()

#         Q = (self.util_counts + self.prior)
#         Q /= Q.sum()
        
#         M = (P + Q) / 2
        
#         return 0.5 * (sum(P * np.log2(P / M)) + 
#                   sum(Q * np.log2(Q / M)))


#     def calc_all_group_query_loss(self):
#         return ((np.abs(self.base_counts - self.util_counts) /
#                  self.data.counts.values) * 100).mean()
    
    
#     def calc_all_group_abs_pct_moved(self):
#         return ((np.abs(self.base_counts - self.util_counts).sum()) /
#                 self.tot_pop) * 100


#     def calc_all_group_marketer(self):
#         """
#         Calculates the overall marketer risk for the given policy.
#         """
        
#         mark_risk = (np.nan_to_num(self.base_counts[self.risk_counts >= self.k] / 
#             self.risk_counts[self.risk_counts >= self.k])
#         .sum() / 
#         self.tot_pop)
        
#         return mark_risk


#     def calc_all_group_prop_suppressed(self):

#         return ((self.risk_counts < self.k) * self.base_counts).sum() / self.tot_pop


#     def get_overall_measures(self):

#         self.ans = pd.DataFrame({'k' : self.k,
#          'suppression_prop_threshold' : self.sup_prop,
#          'grouped_by' : self.group_var},
#          index = [0])

#         # JS divergence
#         self.ans['js_div'] = self.calc_all_group_js_div()

#         # query loss
#         self.ans['query_loss'] = self.calc_all_group_query_loss()

#         # % moved
#         self.ans['abs_pct_moved'] = self.calc_all_group_abs_pct_moved()

#         # % suppressed
#         self.ans['prop_suppressed'] = self.calc_all_group_prop_suppressed()

#         # marketer risk
#         self.ans['marketer'] = self.calc_all_group_marketer()


#     def get_inequality(self):

#         self.ans['js_div_inequality'] = gini(self.all_policy_details['js_div'].values)
#         self.ans['query_loss_inequality'] = gini(self.all_policy_details['query_loss'].values)
#         self.ans['abs_pct_moved_inequality'] = gini(self.all_policy_details['abs_pct_moved'].values)
#         self.ans['prop_suppr_inequality'] = gini(self.all_policy_details['prop_suppressed'].values)
#         self.ans['marketer_inequality'] = gini(self.all_policy_details['marketer'].values)


#     def process_kmin_for_measure(self, util_measure):

#         # get each group's best k-minimal generalization according to utility measure
#         self.get_all_policy_details(util_measure)

#         # add each group's generalization names
#         self.add_names_to_frame(self.all_policy_details)

#         # get group measures from group-specific k-minimal generalizations
#         #self.get_group_measures()

#         # get all util and risk count values for overall measures
#         self.get_all_counts()

#         # calculate overall measures
#         self.get_overall_measures()

#         # calculate inequality
#         self.get_inequality()
