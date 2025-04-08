"""
k-anonymize dataset with Mondrian algorithm.

Receives as input a frequency table for each of the raw
quasi-identifying attributes considered. The frequency 
must be contained in a column labeled 'counts'.

This implementation support the original Mondrian,
where semantics may be broken (e.g., assuming ZIP codes,
that differ by one digit are adjacent to each other),
and a more sophisticated Mondrian that allows specified attributes
to be split according to the provided generalization hierarchies. 
Moreover, this implementation reduces compute time by distributing
the algorithm across the number of cores specified.
"""

## Libraries
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool


## Helper functions
def Mondrian_anonymizer(df, k, feature_columns,
                        sensitive_column, categorical,
                        hierarchical=None, hierarchies=None,
                        geo_hierarchical=None, geo_hierarchies=None,
                        prioritized_var=None, not_suppressed=None,
                        n_cores = 1):
    
    # check specificed columns
    if (sensitive_column not in df.columns) and (sensitive_column is not None):
        raise ValueError("No Such Sensitive Column")

    for fcolumn in feature_columns:
        if fcolumn not in df.columns:
            raise ValueError("No Such Feature Column :"+fcolumn)
    
    if hierarchical is not None:
        for fcolumn in hierarchical:
            if fcolumn not in df.columns:
                raise ValueError("No Such Feature Column :"+fcolumn)
    
    if geo_hierarchical is not None:
        for fcolumn in geo_hierarchical:
            if fcolumn not in df.columns:
                raise ValueError("No Such Feature Column :"+fcolumn)
    
    # initialize object
    anon = Mondrian(df = df,
                    k = k,
                    l = None,
                    feature_columns = feature_columns,
                    sensitive_column = sensitive_column,
                    categorical = categorical,
                    hierarchical=hierarchical,
                    hierarchies=hierarchies,
                    geo_hierarchical=geo_hierarchical,
                    geo_hierarchies=geo_hierarchies,
                    prioritized_var=prioritized_var,
                    not_suppressed=not_suppressed)
    
    # initialize generalization scaling
    anon.get_full_span()
    
    # new equivalence class assignments
    full_partitions = anon.partition_dataset(n_cores = n_cores)
    
    # make assignments
    anon.assign_equivalence_class(full_partitions)
    
    # return input df with equivalence class assignments
    return anon.df


def Mondrian_distribute(df, k, l, feature_columns, categorical, sensitive_column,
                 hierarchical, hierarchies, max_level, scale, geo_hierarchical,
                 geo_hierarchies, max_geo_level, partition):
    
    # subset of original dataset
    ndf = df.loc[partition, :].copy()

    # new Mondrian object
    obj = Mondrian_sub(df = ndf, k=k, l=l, feature_columns=feature_columns,
                       categorical=categorical, sensitive_column=sensitive_column,
                       hierarchical=hierarchical, hierarchies=hierarchies,
                       max_level=max_level, scale=scale, geo_hierarchical=geo_hierarchical,
                       geo_hierarchies=geo_hierarchies, max_geo_level=max_geo_level)

    # partition
    return obj.partition_subset()


def Mondrian_initialize(df, target_k, min_k, feature_columns, categorical,
                 hierarchical, hierarchies, max_level, scale, geo_hierarchical,
                 geo_hierarchies, max_geo_level, group_cores, group_partitions, group_value):
    
    # restrict dataset to group's partition
    partition = group_partitions[group_value]

    # subset of original dataset
    ndf = df.loc[partition, :]

    print('Group:', group_value)

    # new Mondrian object
    obj = Mondrian_fair_sub(df = ndf, target_k=target_k, min_k=min_k, feature_columns=feature_columns,
                        categorical=categorical,
                        hierarchical=hierarchical, hierarchies=hierarchies,
                        max_level=max_level, scale=scale, geo_hierarchical=geo_hierarchical,
                        geo_hierarchies=geo_hierarchies, max_geo_level=max_geo_level,
                        group_value=group_value, n_group_cores=group_cores)

    # partition on group's subset
    obj.parallelize_partitions()

    # calculate utility
    obj.calc_utility()

    return obj


def Mondrian_reiterate(k, obj_dict, group_value):

    obj = obj_dict[group_value]

    # set new value of k
    obj.set_new_k(k)

    # partition per new k value
    obj.parallelize_partitions()

    # calculate utility
    obj.calc_utility()

    #print('Done: {}, k={}'.format(group_value, k))


# def Mondrian_group_helper(obj, partitions):

#     finished_partitions = []

#     print('Partitions1:', partitions)
        
#     # loop through partitions
#     while len(partitions) > 0:
        
#         # choose single partition
#         partition = partitions.pop(0)
        
#         # if multiple bins (splittable)
#         if len(partition) > 1:
            
#             # get spans to prioritize attribute splitting
#             spans = obj.get_spans(partition)
            
#             # iterate through attributes
#             for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                
#                 # span=0 means partition is not splittable on column
#                 if span == 0:
#                     if column in obj.hierarchical:
#                         obj.df.loc[partition, column + '_level'] = obj.max_level[column]
#                     elif column in obj.geo_hierarchical:
#                         obj.df.loc[partition, column + '_level'] = obj.max_geo_level[column]
#                     continue
                
#                 # split partition on attribute
#                 proposed_partitions = obj.split(partition, column)
                
#                 # if attribute cannot be split
#                 if proposed_partitions is None:
#                     continue
                
#                 # if partition can be successfully split on attribute,
#                 # extend partitions with multiple bins
#                 for pp in proposed_partitions:
#                     if len(pp) > 1:
#                         partitions.append(pp)
#                     else:
#                         finished_partitions.append(pp)
                
#                 # change generalization level
#                 if obj.change_hier or obj.change_geo_hier:
#                     obj.df.loc[partition, column + '_level'] = obj.new_level
                
#                 break
                
#             # if partition cannot be split further
#             else:
#                 finished_partitions.append(partition)
        
#         # if partition contains single bin
#         else:
#             finished_partitions.append(partition)

#     return finished_partitions


## Mondrian classes
class Mondrian_sub:

    def __init__(self, df, k, l, feature_columns, sensitive_column, categorical,
                 hierarchical, hierarchies, max_level, scale, geo_hierarchical,
                 geo_hierarchies, max_geo_level):

        self.df = df.copy()
        self.k = k
        self.l = l
        self.feature_columns = feature_columns
        self.sensitive_column = sensitive_column
        self.categorical = categorical
        self.hierarchical = hierarchical
        self.hierarchies = hierarchies
        self.max_level = max_level
        self.scale = scale
        self.geo_hierarchical = geo_hierarchical
        self.geo_hierarchies = geo_hierarchies
        self.max_geo_level = max_geo_level


    def get_spans(self, partition):

        spans = {}
        for column in self.feature_columns:
            
            if column in self.categorical:
                span = len(self.df[column][partition].unique())
                
                    
            else:
                span = self.df[column][partition].max() - self.df[column][partition].min() + 1

            # if only single value
            if span == 1:
                span = 0
                
            if self.scale is not None:
                span = span/self.scale[column]
            else:
                print('Creating scale')
                
            spans[column] = span
            
        return spans

    
    def hierarchical_split(self, partition, column):
        
        proposed_partitions = []
        dfp = self.df[column][partition].copy()
        hierarchy = self.hierarchies[column]
        
        # current generalization level
        self.new_level = self.df.loc[partition, column + '_level'].values[0]

        while len(proposed_partitions) < 2:
            proposed_partitions = []
            
            # step down hierarchy
            self.new_level += 1
            
            # if exceeded hierarchy
            if self.new_level > self.max_level[column]:
                return None
            
            # find non-empty partitions for new generalization level
            for new_category in hierarchy[self.new_level]:

                pp = dfp.index[dfp.isin(new_category)]

                if len(pp) > 0:
                    proposed_partitions.append(pp)

            # if new level can be applied without splitting column
            if len(proposed_partitions) == 1:
                self.df.loc[partition, column + '_level'] = self.new_level 
        
        # k-anonymity check
        for pp in proposed_partitions:
            if not self.is_k_anonymous(pp):
                return None
        
        # return new partitions if all checks passed
        self.change_hier = True
        return proposed_partitions
    
    
    def geo_hierarchical_split(self, partition, column):
        
        proposed_partitions = []
        dfp = self.df[column][partition].copy()
        
        # current generalization level
        self.new_level = self.df.loc[partition, column + '_level'].values[0]

        while len(proposed_partitions) < 2:
            proposed_partitions = []
            
            # step down hierarchy
            self.new_level += 1
            
            # if exceeded hierarchy
            if self.new_level > self.max_geo_level[column]:
                return None
            
            # new geo code string generalization
            num_digits = self.geo_hierarchies[column][self.new_level]

            dfpg = dfp.str[:num_digits]
            
            # find non-empty partitions for new generalization level
            for geo_code in dfpg.unique():

                pp = dfpg.index[dfpg == geo_code]

                if len(pp) > 0:
                    proposed_partitions.append(pp)

            # if new level can be applied without splitting column
            if len(proposed_partitions) == 1:
                self.df.loc[partition, column + '_level'] = self.new_level        
        
        # k-anonymity check
        for pp in proposed_partitions:
            if not self.is_k_anonymous(pp):
                return None
        
        # return new partitions if all checks passed
        self.change_geo_hier = True
        return proposed_partitions
    
    
    def categorical_split(self, partition, column):
        
        dfp = self.df[column][partition].copy()
        values = dfp.unique()
        
        # not splittable if single value
        if len(values) == 1:
            return None
        else:
            lv = set(values[:len(values)//2])
            rv = set(values[len(values)//2:])

            dlv = dfp.index[dfp.isin(lv)]
            drv = dfp.index[dfp.isin(rv)]

            for pp in [dlv, drv]:
                
                # check for non-empty partition
                if len(pp) > 0:
                    
                    #k-anonymity check
                    if not self.is_k_anonymous(pp):
                        return None
                
                else:
                    return None
                
        # return new partitions if all checks passed
        return [dlv, drv]
    
    
    def numerical_split(self, partition, column):
        
        # calculate median from frequency table
        dfp = self.df.loc[partition, [column, 'counts']].sort_values(column)['counts'].copy()
        cum_pct = dfp.cumsum() / dfp.sum()
        
        median_idx = np.where(cum_pct >= 0.5)[0][0]
        
        # split on median
        dfl = dfp.index[:median_idx]
        dfr = dfp.index[median_idx:]
        
        # if skewed to first bin - IS THIS APPROPRIATE?
        if (len(dfl) == 0) and (len(dfr) > 1):
            dfl = dfp.index[:1]
            dfr = dfp.index[1:]

        for pp in [dfl, dfr]:
            
            # check for non-empty partition
            if len(pp) > 0:
                
                # k-anonymity check
                if not self.is_k_anonymous(pp):
                    return None
                
            else:
                return None
        
        # return new partitions if all checks passed
        return [dfl, dfr]
        

    def split(self, partition, column):
        
        self.change_hier = False
        self.change_geo_hier = False
        
        # generalization hierarchy without geographical specification
        if column in self.hierarchical:
            return self.hierarchical_split(partition, column)
        
        # generalization hierarchy with geographical specification
        elif column in self.geo_hierarchical:
            return self.geo_hierarchical_split(partition, column)
        
        # no generalization hierarchy
        else:
            
            # categorical attribute
            if column in self.categorical:
                return self.categorical_split(partition, column)
            
            # numerical attribute
            else:
                return self.numerical_split(partition, column)


    def is_k_anonymous(self, partition):

        if self.df['counts'][partition].sum() < self.k:
            return False
        else:
            return True


    def l_diversity(self, df, partition, column):
        """ FIX """
        return len(df[column][partition].unique())


    def is_l_diverse(self, df, partition, sensitive_column, l):
        """ FIX """
        return l_diversity(df, partition, sensitive_column) >= l


    # @PARAMS - partition_dataset()
    # df - pandas dataframe
    # feature_column - list of column names along which to partitions the dataset
    # scale - column spans


    def partition_subset(self, partitions = None, return_df = True):

        finished_partitions = []
        
        # initialize with all tuples
        if partitions is None:
            partitions = [self.df.index]
        
        # loop through partitions
        while partitions:
            
            # choose single partition
            partition = partitions.pop(0)
            
            # if multiple bins (splittable)
            if len(partition) > 1:
                
                # get spans to prioritize attribute splitting
                spans = self.get_spans(partition)
                
                # iterate through attributes
                for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                    
                    # span=0 means partition is not splittable on column
                    if span == 0:
                        if column in self.hierarchical:
                            self.df.loc[partition, column + '_level'] = self.max_level[column]
                        elif column in self.geo_hierarchical:
                            self.df.loc[partition, column + '_level'] = self.max_geo_level[column]
                        continue
                    
                    # split partition on attribute
                    proposed_partitions = self.split(partition, column)
                    
                    # if attribute cannot be split
                    if proposed_partitions is None:
                        continue
                    
                    # if partition can be successfully split on attribute,
                    # extend partitions with multiple bins
                    for pp in proposed_partitions:
                        if len(pp) > 1:
                            partitions.append(pp)
                        else:
                            finished_partitions.append(pp)
                    
                    # change generalization level
                    if self.change_hier or self.change_geo_hier:
                        self.df.loc[partition, column + '_level'] = self.new_level
                    
                    break
                    
                # if partition cannot be split further
                else:
                    finished_partitions.append(partition)
            
            # if partition contains single bin
            else:
                finished_partitions.append(partition)

        if return_df:
            return [finished_partitions, self.df.copy()]
        else:
            return finished_partitions



class Mondrian(Mondrian_sub):
    
    def __init__(self, df, k, l, feature_columns, categorical,
                 sensitive_column=None, hierarchical=None, hierarchies=None,
                 geo_hierarchical=None, geo_hierarchies=None, prioritized_var=None,
                 not_suppressed=None):
        
        self.df = df.copy()
        self.k = k
        self.l = l
        self.feature_columns = feature_columns
        self.categorical = categorical
        self.sensitive_column = sensitive_column
        self.hierarchical = hierarchical
        self.hierarchies = hierarchies
        self.max_level = {}
        self.geo_hierarchical = geo_hierarchical
        self.geo_hierarchies = geo_hierarchies
        self.max_geo_level = {}
        self.prioritized_var = prioritized_var
        self.not_suppressed = not_suppressed

        if hierarchical is not None:            
            if hierarchies is None:
                raise ValueError("Need to supply generalization hierarchies.")                     
            for hier_col in list(hierarchical):
                self.df[hier_col + '_level'] = 0
                self.max_level[hier_col] = max(self.hierarchies[hier_col].keys())
        else:
            self.hierarchical = set([])
            
        if geo_hierarchical is not None:
            if geo_hierarchies is None:
                raise ValueError("Need to supply geographic column's generalization hierarchies.")
            for hier_col in list(geo_hierarchical):
                self.df[hier_col + '_level'] = 0
                self.max_geo_level[hier_col] = max(self.geo_hierarchies[hier_col].keys())
        else:
            self.geo_hierarchical = set([])

        if (self.prioritized_var is not None) & (self.not_suppressed is not None):
            raise ValueError("Choose prioritized_var or not_suppressed. Choosing both may cause unintended errors.")

        # get scale
        self.scale = None
        self.get_full_span()


    def get_full_span(self):
        
        for name in self.feature_columns:
            if name not in self.categorical:
                self.df[name] = pd.to_numeric(self.df[name])
        
        self.scale = self.get_spans(self.df.index)


    def partition_dataset(self, n_cores):

        # non prioritized partitions
        if (self.prioritized_var is None) & (self.not_suppressed is None):

            # not distributed
            if n_cores <= 1:
                return self.partition_subset(return_df=False)

            # distributed
            else:
                return self.distributed_partitions(n_cores=n_cores)

        # partitions preventing attribute suppression
        elif self.not_suppressed is not None:

            print('{} variable will not be suppressed'.format(self.not_suppressed))

            prioritized_partitions = self.initialize_not_suppressed_partitions()

            # not distributed
            if n_cores <= 1:
                return self.partition_subset(partitions=prioritized_partitions, return_df=False)

            # distributed
            else:
                return self.distributed_partitions(n_cores=n_cores, prioritized_partitions=prioritized_partitions)

        # prioritized partitions
        else:

            print('{} variable will be prioritized_var'.format(self.prioritized_var))

            prioritized_partitions = self.initialize_prioritized_partitions()

            # not distributed
            if n_cores <= 1:
                return self.partition_subset(partitions=prioritized_partitions, return_df=False)

            # distributed
            else:
                return self.distributed_partitions(n_cores=n_cores, prioritized_partitions=prioritized_partitions)


    def initialize_distribution(self, n_cores, partitions=None):
        """
        partition dataset until num partitions >= cores
        """

        finished_partitions = []
        
        # initialize with all tuples, unless already initialized
        if partitions is None:
            partitions = [self.df.index]
        
        # loop through partitions
        while partitions and (len(partitions) < n_cores):
            
            # choose single partition
            partition = partitions.pop(0)
            
            # if multiple bins (splittable)
            if len(partition) > 1:
                
                # get spans to prioritize attribute splitting
                spans = self.get_spans(partition)
                
                # iterate through attributes
                for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                    
                    # span=0 means partition is not splittable on column
                    if span == 0:
                        continue
                    
                    # split partition on attribute
                    proposed_partitions = self.split(partition, column)
                    
                    # if attribute cannot be split
                    if proposed_partitions is None:
                        continue
                    
                    # if partition can be successfully split on attribute,
                    # extend partitions with multiple bins
                    for pp in proposed_partitions:
                        if len(pp) > 1:
                            partitions.append(pp)
                        else:
                            finished_partitions.append(pp)
                    
                    # change generalization level
                    if self.change_hier or self.change_geo_hier:
                        self.df.loc[partition, column + '_level'] = self.new_level
                    
                    break
                    
                # if partition cannot be split further
                else:
                    finished_partitions.append(partition)
            
            # if partition contains single bin
            else:
                finished_partitions.append(partition)

        return finished_partitions, partitions


    def initialize_prioritized_partitions(self):
        """
        Performs first split(s) on the prioritized variable, so as to
        maximize the granularity at which the prioritized variable
        is preserved in the dataset.
        """

        # split starting from the bottom of the hierarchy
        if self.prioritized_var in self.hierarchical:

            return self.priority_partition_hierarchical()

        elif self.prioritized_var in self.geo_hierarchical:

            return self.priority_partition_geographical()

        else:
            raise ValueError('Add normal categorical and numerical presplitting!')


    def initialize_not_suppressed_partitions(self):
        """
        Split on first level of generalization hierarchy (does not work for geocode strings).
        """

        proposed_partitions = []
        feasible = True

        # get non_suppressed column values
        dfp = self.df[self.not_suppressed].copy()

        # not suppressed column hierarchy
        hierarchy = self.hierarchies[self.not_suppressed]

        # split on first level
        level = list(hierarchy.keys())[0]

        for new_category in hierarchy[level]:

                pp = dfp.index[dfp.isin(new_category)]

                if len(pp) > 0:

                    if not self.is_k_anonymous(pp):
                        feasible = False
                        break

                    proposed_partitions.append(pp)

        if feasible:
            self.df.loc[:, self.not_suppressed + '_level'] = level # assign hierarchy level
            return proposed_partitions
        else:
            raise ValueError('Cannot make initial split on {} column and meet k threshold')
            self.df.loc[:, self.not_suppressed + '_level'] = -1 # assign hierarchy level
            return [self.df.index]


    def priority_partition_hierarchical(self):
        """
        Split as much as possible on self.prioritized var.
        """

        finished_partitions = []
        
        # initialize with all tuples
        partitions = [self.df.index]

        while partitions:

            # choose single partition
            partition = partitions.pop(0)
            
            # if multiple bins (splittable)
            if len(partition) > 1:

                # if multiple variable values
                if len(self.df.loc[partition, self.prioritized_var].unique()) == 1:
                    self.df.loc[partition, self.prioritized_var + '_level'] = self.max_level[self.prioritized_var]
                    finished_partitions.append(partition)

                else:
                    # bottom-up split search
                    proposed_partitions, max_reached = self.bottom_up_hierarchical_split(partition)

                    # if maximum granularity is already reached
                    if max_reached:
                        finished_partitions.extend(proposed_partitions)

                    else:
                        # if partition could not be split
                        if len(proposed_partitions) == 1:
                            finished_partitions.append(partition)

                        # if it could be split
                        else:

                            for pp in proposed_partitions:
                                # filter out partitions with single variable value
                                if len(self.df.loc[pp, self.prioritized_var].unique()) > 1:
                                    partitions.append(pp)
                                else:
                                    self.df.loc[pp, self.prioritized_var + '_level'] = self.max_level[self.prioritized_var]
                                    finished_partitions.append(pp)

            # if partition contains single bin
            else:
                finished_partitions.append(partition)

        return finished_partitions


    def priority_partition_geographical(self):
        """
        Split as much as possible on self.prioritized var.
        """

        finished_partitions = []
        
        # initialize with all tuples
        partitions = [self.df.index]

        while partitions:

            # choose single partition
            partition = partitions.pop(0)
            
            # if multiple bins (splittable)
            if len(partition) > 1:

                # if multiple variable values
                if len(self.df.loc[partition, self.prioritized_var].unique()) == 1:
                    self.df.loc[partition, self.prioritized_var + '_level'] = self.max_level[self.prioritized_var]
                    finished_partitions.append(partition)

                else:
                    # bottom-up split search
                    proposed_partitions, max_reached = self.bottom_up_geo_hierarchical_split(partition)

                    # if maximum granularity is already reached
                    if max_reached:
                        finished_partitions.extend(proposed_partitions)

                    else:
                        # if partition could not be split
                        if len(proposed_partitions) == 1:
                            finished_partitions.append(partition)

                        # if it could be split
                        else:

                            for pp in proposed_partitions:
                                # filter out partitions with single variable value
                                if len(self.df.loc[pp, self.prioritized_var].unique()) > 1:
                                    partitions.append(pp)
                                else:
                                    self.df.loc[pp, self.prioritized_var + '_level'] = self.max_level[self.prioritized_var]
                                    finished_partitions.append(pp)

            # if partition contains single bin
            else:
                finished_partitions.append(partition)

        return finished_partitions


    def bottom_up_hierarchical_split(self, partition):
        
        proposed_partitions = []
        dfp = self.df[self.prioritized_var][partition].copy()
        hierarchy = self.hierarchies[self.prioritized_var]

        # start splitting at bottom level of hierarchy - assumes hierarchy is correctly ordered
        for level in list(hierarchy.keys())[::-1]:

            continue_outer_loop = False

            for new_category in hierarchy[level]:

                pp = dfp.index[dfp.isin(new_category)]

                if len(pp) > 0:

                    if self.is_k_anonymous(pp):
                        proposed_partitions.append(pp)
                    else:
                        continue_outer_loop = True
                        break

            if not continue_outer_loop:
                break

        # assign hierarchy level
        self.df.loc[partition, self.prioritized_var + '_level'] = level

        if len(proposed_partitions) == 0:
            print('Bottom up error!')

        if level == self.max_level[self.prioritized_var]:
            max_reached = True
        else:
            max_reached = False

        return proposed_partitions, max_reached


    def bottom_up_geo_hierarchical_split(self, partition):
        """
        NEED TEST
        """

        proposed_partitions = []
        dfp = self.df[column][partition].copy()
        hierarchy = self.geo_hierarchies[self.prioritized_var]

         # start splitting at bottom level of hierarchy - assumes hierarchy is correctly ordered
        for num_digits in list(hierarchy.keys())[::-1]:

            continue_outer_loop = False

            dfpg = dfp.str[:num_digits]

            # find non-empty partitions for new generalization level
            for geo_code in dfpg.unique():

                pp = dfpg.index[dfpg == geo_code]

                if len(pp) > 0:
                    
                    if self.is_k_anonymous(pp):
                        proposed_partitions.append(pp)
                    else:
                        continue_outer_loop = True
                        break

            if not continue_outer_loop:
                break

        # assign hierarchy level
        self.df.loc[partition, self.prioritized_var + '_level'] = level

        if len(proposed_partitions) == 0:
            print('Bottom up geo error!')

        if num_digits == self.max_geo_level[self.prioritized_var]:
            max_reached = True
        else:
            max_reached = False

        return proposed_partitions, max_reached
        

    def distributed_partitions(self, n_cores, prioritized_partitions=None):


        # initialize partitions
        if prioritized_partitions is None:

            finished_partitions, partitions = self.initialize_distribution(n_cores)
            feature_columns = self.feature_columns

        else:

            finished_partitions, partitions = self.initialize_distribution(n_cores, partitions=prioritized_partitions)
            # reduce feature columns
            feature_columns = self.feature_columns.copy()

            if self.prioritized_var is not None:
                feature_columns.remove(self.prioritized_var)

        # distribute remaining partitions
        if len(partitions) > 0:

            pool = Pool(processes = n_cores)

            results = pool.map_async(partial(Mondrian_distribute,
                                                            self.df,
                                                            self.k,
                                                            self.l,
                                                            feature_columns,
                                                            self.categorical,
                                                            self.sensitive_column,
                                                            self.hierarchical,
                                                            self.hierarchies,
                                                            self.max_level,
                                                            self.scale,
                                                            self.geo_hierarchical,
                                                            self.geo_hierarchies,
                                                            self.max_geo_level),
                                                    partitions)

            # separate and concatenate results
            if len(finished_partitions) > 0:
                #print(finished_partitions)
                new_df = [self.df.loc[sum(finished_partitions, []), :].copy()]
            else:
                new_df = []

            for result in results.get():
                finished_partitions.extend(result[0])
                new_df.append(result[1])

            # close object
            pool.close()
            pool.join()

            # concatenate distributed_dfs to preserve hierarchical generalization assignments
            self.df = pd.concat(new_df, ignore_index=False).sort_index()

            return finished_partitions

    
    def assign_equivalence_class(self, full_partitions):
        """
        Assigns numeric label specifying equivalence class.
        """
        
        equiv_class = np.zeros(len(self.df)).astype(int) - 1
        
        for i in range(len(full_partitions)):
            partition = full_partitions[i]
            equiv_class[partition] = i
        
        self.df['equivalence_class'] = equiv_class



class Mondrian_fair_sub(Mondrian_sub):

    def __init__(self, df, target_k, min_k, feature_columns, categorical,
                 hierarchical, hierarchies, max_level, scale, geo_hierarchical,
                 geo_hierarchies, max_geo_level, group_value, n_group_cores):

        self.df = df.copy()
        self.k = target_k
        self.min_k = min_k
        self.target_k = target_k
        self.feature_columns = feature_columns
        self.categorical = categorical
        self.hierarchical = hierarchical
        self.hierarchies = hierarchies
        self.max_level = max_level
        self.scale = scale
        self.geo_hierarchical = geo_hierarchical
        self.geo_hierarchies = geo_hierarchies
        self.max_geo_level = max_geo_level
        self.group_value = group_value
        self.n_group_cores = n_group_cores
        self.tot_pop = self.df.counts.sum()
        self.partition_dict = {}
        self.utility_dict = {}


    def set_new_k(self, k):
        self.k = k


    def set_utility_threshold(self, value):
        self.utility_threshold = value


    def initialize_partitions_for_distribution(self, partitions):
        """
        partition dataset until num partitions >= n_group_cores
        """
        
        # loop through partitions
        while partitions and (len(partitions) < self.n_group_cores):
            
            # choose single partition
            partition = partitions.pop(0)
            
            # if multiple bins (splittable)
            if len(partition) > 1:
                
                # get spans to prioritize attribute splitting
                spans = self.get_spans(partition)
                
                # iterate through attributes
                for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                    
                    # span=0 means partition is not splittable on column
                    if span == 0:
                        continue
                    
                    # split partition on attribute
                    proposed_partitions = self.split(partition, column)
                    
                    # if attribute cannot be split
                    if proposed_partitions is None:
                        continue
                    
                    # if partition can be successfully split on attribute,
                    # extend partitions with multiple bins
                    for pp in proposed_partitions:
                        if len(pp) > 1:
                            partitions.append(pp)
                        else:
                            self.finished_partitions.append(pp)
                    
                    # change generalization level
                    if self.change_hier or self.change_geo_hier:
                        self.df.loc[partition, column + '_level'] = self.new_level
                    
                    break
                    
                # if partition cannot be split further
                else:
                    self.finished_partitions.append(partition)
            
            # if partition contains single bin
            else:
                self.finished_partitions.append(partition)

        return partitions


    def get_starting_partitions(self):

        current_k = self.k

        # if no previous partition set is available (ie when doing initial partitions)
        if len(self.partition_dict) == 0:
            return [self.df.index]

        # find k value closest to current k, that is still greater than current k
        # default is starting k value
        else:

            best_k = self.target_k

            for k in self.partition_dict.keys():

                if (k > current_k) & (k < best_k):

                    best_k = k

        # return partitions corresponding to best_k
        return self.partition_dict[best_k]


    def parallelize_partitions(self):

        if self.k <= 1:

            self.partition_dict[self.k] = list(self.df.index.values.reshape(-1, 1))

        else:

            # get starting partitions - closest k value > current k value
            start_partitions = self.get_starting_partitions()

            # initialize partitions until start_partitions >= self.n_group_cores
            if len(start_partitions) < self.n_group_cores:
                start_partitions = self.initialize_partitions_for_distribution(start_partitions)

            # initialize pool object
            pool = Pool(processes = self.n_group_cores)

            # partition
            new_partitions = pool.map_async(self.partition_subset_parallel, start_partitions).get()

            # save results
            finished_partitions = []
            for new_set in new_partitions:
                #print(new_set)
                finished_partitions.extend(new_set)

            self.partition_dict[self.k] = finished_partitions

            # close pool object
            pool.close()
            pool.join()


    def partition_subset_parallel(self, partitions):

        finished_partitions = []

        partitions = [partitions]
        
        # loop through partitions
        while partitions:
            
            # choose single partition
            partition = partitions.pop(0)
            
            # if multiple bins (splittable)
            if len(partition) > 1:
                
                # get spans to prioritize attribute splitting
                spans = self.get_spans(partition)
                
                # iterate through attributes
                for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                    
                    # span=0 means partition is not splittable on column
                    if span == 0:
                        if column in self.hierarchical:
                            self.df.loc[partition, column + '_level'] = self.max_level[column]
                        elif column in self.geo_hierarchical:
                            self.df.loc[partition, column + '_level'] = self.max_geo_level[column]
                        continue
                    
                    # split partition on attribute
                    proposed_partitions = self.split(partition, column)
                    
                    # if attribute cannot be split
                    if proposed_partitions is None:
                        continue
                    
                    # if partition can be successfully split on attribute,
                    # extend partitions with multiple bins
                    for pp in proposed_partitions:
                        if len(pp) > 1:
                            partitions.append(pp)
                        else:
                            finished_partitions.append(pp)
                    
                    # change generalization level
                    if self.change_hier or self.change_geo_hier:
                        self.df.loc[partition, column + '_level'] = self.new_level
                    
                    break
                    
                # if partition cannot be split further
                else:
                    finished_partitions.append(partition)
            
            # if partition contains single bin
            else:
                finished_partitions.append(partition)

        return finished_partitions


    # def partition_group(self, partitions = None):

    #     self.finished_partitions = []
        
    #     # initialize with all tuples
    #     if partitions is None:
    #         partitions = [self.df.index]
        
    #     # loop through partitions
    #     while partitions:
            
    #         # choose single partition
    #         partition = partitions.pop(0)
            
    #         # if multiple bins (splittable)
    #         if len(partition) > 1:
                
    #             # get spans to prioritize attribute splitting
    #             spans = self.get_spans(partition)
                
    #             # iterate through attributes
    #             for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                    
    #                 # span=0 means partition is not splittable on column
    #                 if span == 0:
    #                     if column in self.hierarchical:
    #                         self.df.loc[partition, column + '_level'] = self.max_level[column]
    #                     elif column in self.geo_hierarchical:
    #                         self.df.loc[partition, column + '_level'] = self.max_geo_level[column]
    #                     continue
                    
    #                 # split partition on attribute
    #                 proposed_partitions = self.split(partition, column)
                    
    #                 # if attribute cannot be split
    #                 if proposed_partitions is None:
    #                     continue
                    
    #                 # if partition can be successfully split on attribute,
    #                 # extend partitions with multiple bins
    #                 for pp in proposed_partitions:
    #                     if len(pp) > 1:
    #                         partitions.append(pp)
    #                     else:
    #                         self.finished_partitions.append(pp)
                    
    #                 # change generalization level
    #                 if self.change_hier or self.change_geo_hier:
    #                     self.df.loc[partition, column + '_level'] = self.new_level
                    
    #                 break
                    
    #             # if partition cannot be split further
    #             else:
    #                 self.finished_partitions.append(partition)
            
    #         # if partition contains single bin
    #         else:
    #             self.finished_partitions.append(partition)


    def full_process_at_k(self, k):

        #print('Starting k={}'.format(k))

        # set new k value
        self.set_new_k(k)

        # partition dataset at new k value, if partitions do not already exist
        if self.k not in self.partition_dict.keys():
            self.parallelize_partitions()
        # else:
        #     print('Skip partition')

        # calculate utility
        self.calc_utility()


    def meet_utility_threshold(self):

        utility = self.utility_dict[self.k]

        if utility <= self.utility_threshold:
            return True
        else:
            return False


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


    def k_search(self):

        """
        Find maximum k that meets utility threshold
        """


        if self.meet_utility_threshold():
            new_k = self.target_k

        else:
            new_k = self.binary_search(low=self.min_k, high=self.target_k)

        self.final_process(new_k)


    def final_process(self, k):

        self.set_new_k(k)
        self.final_k = k
        self.final_utility = self.utility_dict[k]
        self.final_partitions = self.partition_dict[k]


    def calc_marketer_risk(self):

        # assign equivalence classes
        self.set_new_k(self.final_k)

        # calculate initial
        self.calc_initial_marketer_risk()

        # calculate final
        self.calc_final_marketer_risk()


    def calc_initial_marketer_risk(self):

        self.initial_marketer_risk = len(self.df) / self.tot_pop


    def calc_final_marketer_risk(self):

        self.final_marketer_risk = len(self.final_partitions) / self.tot_pop


    def assign_equivalence_class(self):

        if self.k <= 1:

            self.df['equivalence_class'] = self.df.index.values

        else:

            finished_partitions = self.partition_dict[self.k]

            equiv_class = np.zeros(max(self.df.index) + 1).astype(int) - 1
        
            for i in range(len(finished_partitions)):
                partition = finished_partitions[i]
                equiv_class[partition] = i

            equiv_class = equiv_class[equiv_class > -1]

            self.df['equivalence_class'] = equiv_class


    def calc_JS_div(self, prior = 1e-20):

        # define distributions (add weak prior)
        P = self.df.counts.values + prior
        Q = ((self.df.groupby('equivalence_class', sort=False)
                ['counts']
                .transform('mean')).values + 
             prior)

        # convert to probabilities
        P /= P.sum()
        Q /= Q.sum()

        # define M for JS divergence
        M = (P + Q) / 2

        # KL divergence
        #kl_div = sum(P * np.log(P / Q))

        # JS divergence
        js_div = (sum(P * np.log(P / M)) + 
                  sum(Q * np.log(Q / M)))
        
        # save value
        self.utility = 1 - js_div


    def calc_nonuniform_entropy(self):
        """
        Assumes that the probability of suppressed records = # records for the bin / 
        total # records in the dataset. This has the caveat that records who have their
        quasi-identifier suppressed, but not their remaining values, have the same utility
        as those records that are suppressed completely from the dataset.
        """

        P = self.df.counts.values
        Q = (self.df.groupby('equivalence_class', sort=False)
                ['counts']
                .transform('sum')).values

        self.utility = (-np.log2(P / Q) * P).sum() / self.tot_pop
    

    def calc_utility(self):

        if self.k not in self.utility_dict.keys():

            if self.k <= 1:

                self.utility = 0

            else:

                self.assign_equivalence_class()
                self.calc_nonuniform_entropy()

            # store utility
            self.utility_dict[self.k] = self.utility

        # else:
        #     print('Skip utility')



class Mondrian_fairU(Mondrian):
    
    def __init__(self, df, target_k, feature_columns, categorical, group_var,
                 hierarchical=None, hierarchies=None,
                 geo_hierarchical=None, geo_hierarchies=None, minimal_k=1,
                 delta_u=0):
        
        self.df = df.copy()
        self.target_k = target_k
        self.k = target_k
        self.min_k = minimal_k
        self.feature_columns = feature_columns
        self.categorical = categorical
        self.hierarchical = hierarchical
        self.hierarchies = hierarchies
        self.max_level = {}
        self.geo_hierarchical = geo_hierarchical
        self.geo_hierarchies = geo_hierarchies
        self.max_geo_level = {}
        self.group_var = group_var
        self.delta_u = delta_u
        self.group_vals = list(self.df[self.group_var].unique())
        self.tot_pop = self.df.counts.sum()

        if hierarchical is not None:            
            if hierarchies is None:
                raise ValueError("Need to supply generalization hierarchies.")                     
            for hier_col in list(hierarchical):
                self.df[hier_col + '_level'] = 0
                self.max_level[hier_col] = max(self.hierarchies[hier_col].keys())
        else:
            self.hierarchical = set([])
            
        if geo_hierarchical is not None:
            if geo_hierarchies is None:
                raise ValueError("Need to supply geographic column's generalization hierarchies.")
            for hier_col in list(geo_hierarchical):
                self.df[hier_col + '_level'] = 0
                self.max_geo_level[hier_col] = max(self.geo_hierarchies[hier_col].keys())
        else:
            self.geo_hierarchical = set([])

        # get scale
        self.scale = None
        self.get_full_span()


    def is_min_k_anonymous(self, partition):

        if self.df['counts'][partition].sum() < self.min_k:
            return False
        else:
            return True


    def partition_dataset(self, n_cores):

        self.initialize_group_partitions()
        self.fair_partitions(n_cores=n_cores)


    def initialize_group_partitions(self):
        """
        Performs first split(s) on the group variable for which
        group fairness will be calculated.
        """

        self.group_partitions = {}
        dfp = self.df[self.group_var].copy()

        for group_val in self.group_vals:

            pp = dfp.index[dfp == group_val]

            if len(pp) > 0:

                self.group_partitions[group_val] = pp

                if not self.is_k_anonymous(pp):
                    raise ValueError('{} group has < {} records in dataset'.format(group_val, self.target_k))

            else:
                raise ValueError('{} group has < {} records in dataset'.format(group_val, self.target_k))


    def concatenate_partition_indices(self):

        self.all_partitions = []

        for key, obj in self.obj_dict.items():

            self.all_partitions.extend(obj.finished_partitions)


    def calc_overall_JS_div(self, prior = 1e-20):

        # define distributions (add weak prior)
        P = self.df.counts.values + prior
        Q = ((self.df.groupby('equivalence_class', sort=False)
                ['counts']
                .transform('mean')).values + 
             prior)

        # convert to probabilities
        P /= P.sum()
        Q /= Q.sum()

        # define M for JS divergence
        M = (P + Q) / 2

        # KL divergence
        #kl_div = sum(P * np.log(P / Q))

        # JS divergence
        js_div = (sum(P * np.log(P / M)) + 
                  sum(Q * np.log(Q / M)))
        
        # save value
        self.overall_utility = js_div
        self.ans['overall_JS_div'] = js_div
        self.ans['overall_utility'] = 1 - js_div


    def initialize_at_target(self, n_cores):

        # reduce feature columns
        feature_columns = self.feature_columns.copy()
        feature_columns.remove(self.group_var)

        print('----------------Initializing objects----------------')


        self.Mondrian_objs = {}
        for group_val in self.group_vals:
            self.Mondrian_objs[group_val] = Mondrian_initialize(
                                                    self.df,
                                                    self.target_k,
                                                    self.min_k,
                                                    feature_columns,
                                                    self.categorical,
                                                    self.hierarchical,
                                                    self.hierarchies,
                                                    self.max_level,
                                                    self.scale,
                                                    self.geo_hierarchical,
                                                    self.geo_hierarchies,
                                                    self.max_geo_level,
                                                    n_cores,
                                                    self.group_partitions,
                                                    group_val)


    def find_utility_threshold(self):

        max_utility = np.inf
        max_group = ''

        print('---------------Finding maximum utility ---------')
        print('Initial utility:')

        # find maximum utility
        for group, obj in self.Mondrian_objs.items():

            group_util = obj.utility

            print(group, ':', group_util)

            if group_util < max_utility:

                max_utility = group_util
                max_group = group

        # define utility threshold
        self.utility_threshold = max_utility * (1 + self.delta_u)

        print()
        print('Maximum utility:', max_utility)
        print('Group with max:', max_group)
        print('Utility threshold:', self.utility_threshold)


    def set_utility_threshold(self):

        print('---------Setting utility threshold ---------')

        for group, obj in self.Mondrian_objs.items():

            obj.set_utility_threshold(self.utility_threshold)



    def binary_search_for_k(self):

        print('------------Binary search for new k ---------')

        for group, obj in self.Mondrian_objs.items():

            obj.k_search()

            print(group, ':', obj.final_k, ',', obj.final_utility)


    def calc_overall_utility(self):

        overall_utility = (self.ans['nonu_ent'] * self.ans['population']).sum() / self.tot_pop

        self.ans['overall_nonu_ent'] = overall_utility


    def calc_overall_marketer(self):

        # intial
        self.ans['overall_initial_marketer_risk'] = len(self.df) / self.tot_pop

        # final
        self.ans['overall_final_marketer_risk'] = (self.ans['final_marketer_risk'] * self.ans['population']).sum() / self.tot_pop

        # ratio
        self.ans['overall_marketer_ratio'] = self.ans['overall_final_marketer_risk'] / self.ans['overall_initial_marketer_risk']


    def calc_group_risk(self):

        groups = []
        initial_risks = []
        risks = []

        for group, obj in self.Mondrian_objs.items():

            obj.calc_marketer_risk()

            groups.append(group)
            initial_risks.append(obj.initial_marketer_risk)
            risks.append(obj.final_marketer_risk)

        risk_df = pd.DataFrame({
            self.group_var: groups,
            'initial_marketer_risk':initial_risks,
            'final_marketer_risk': risks
            })

        risk_df['marketer_ratio'] = risk_df['final_marketer_risk'] / risk_df['initial_marketer_risk']

        self.ans = self.ans.merge(risk_df, on = self.group_var, how='left')


    def process_output(self):

        # combine group details
        groups = []
        utils = []
        ks = []
        pops = []

        for group, obj in self.Mondrian_objs.items():

            groups.append(group)
            utils.append(obj.final_utility)
            ks.append(obj.final_k)
            pops.append(obj.tot_pop)

        self.ans = pd.DataFrame({
            self.group_var:groups,
            'nonu_ent':utils,
            'k':ks,
            'population':pops
            })

        # group marketer
        self.calc_group_risk()

        # overall utility
        self.calc_overall_utility()

        # overall marketer
        self.calc_overall_marketer()

        # add parameter values
        self.ans['target_k'] = self.target_k
        self.ans['min_k'] = self.min_k
        self.ans['delta_u'] = self.delta_u
        self.ans['utility_threshold'] = self.utility_threshold

    
    def run(self, n_cores):
        self.initialize_group_partitions()
        self.initialize_at_target(n_cores)
        self.find_utility_threshold()
        self.set_utility_threshold()
        self.binary_search_for_k()
        self.process_output()

        print('----------------Done----------------')




