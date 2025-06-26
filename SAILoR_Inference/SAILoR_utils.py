# -*- coding: utf-8 -*-
import argparse
import ast
import itertools
import os
import random
import re
import sys
import time
from collections import defaultdict
from math import floor

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms


# (The top part of the file remains the same)
# ...

class GeneticAlgorithm:
    def __init__(self, nNodes, nRegs, netProps, initialPop=None, initialPopProb=None):
        self.nNodes = nNodes
        self.nRegs = nRegs
        self.netProps = netProps
        self.initialPop = initialPop
        self.initialPopProb = initialPopProb
        self.mutP = 1.0 / self.nNodes
        self.cxP = 0.5
        self.nGens = 100
        self.mu = 100
        self.lambda_ = 200
        self.pMut = 0.5
        self.pCx = 0.5
        self.pSel = 0.5
        self.creator = self.create()
        self.toolbox = self.getToolbox()
        self.population = self.getInitialPop()

    def create(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        return creator

    def getToolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.nNodes)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluation)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutation)
        toolbox.register("select", tools.selNSGA2)
        return toolbox

    def getInitialPop(self):
        if self.initialPop is None:
            return self.toolbox.population(n=self.mu)
        else:
            pop = []
            for i in self.initialPop:
                ind = creator.Individual(i)
                pop.append(ind)
            return pop

    def run(self):
        population = self.toolbox.population(n=self.mu)
        if self.initialPop is not None:
            for i in range(len(self.initialPop)):
                population[i] = creator.Individual(self.initialPop[i])

        for ind in population:
            ind.fitness.values = self.toolbox.evaluate(ind)

        for gen in range(1, self.nGens + 1):
            t1 = time.time()
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=self.cxP, mutpb=self.mutP)

            for ind in offspring:
                ind.fitness.values = self.toolbox.evaluate(ind)

            population = self.toolbox.select(population + offspring, self.mu)
            t2 = time.time()
            print(f"NSGA-II generation {gen}")
            print(f"Generation: {t2 - t1} seconds elapsed")

        return population

    def evaluation(self, individual):
        return self.netProps.getObj1(individual), self.netProps.getObj2(individual)

    def mutation(self, individual):
        indices = np.where(np.array(self.netProps.getRegPotentials()) > 0)[0]
        if len(indices) == 0:
            return individual,
        ind_num = np.random.choice(indices)
        individual[ind_num] = 1 - individual[ind_num]
        return individual,


class ReferenceNetwork:
    def __init__(self, df):
        self.df = df

    def getInDegs(self):
        inDegs = defaultdict(int)
        for index, row in self.df.iterrows():
            inDegs[row[1]] += 1
        return inDegs

    def getOutDegs(self):
        outDegs = defaultdict(int)
        for index, row in self.df.iterrows():
            outDegs[row[0]] += 1
        return outDegs

    def getDegree(self):
        return self.getInDegs(), self.getOutDegs()

    def getEdges(self):
        edges = []
        for index, row in self.df.iterrows():
            edges.append((row[0], row[1]))
        return edges

    def getNodes(self):
        return np.unique(self.df.values)


class NetworkProperties:
    def __init__(self, referenceNets, regWeights, nNodes, geneIndices, geneNames, exactNetworksIndices=[], maxRegs=3):
        self.referenceNets = referenceNets
        self.regWeights = regWeights
        self.nNodes = nNodes
        self.geneIndices = geneIndices
        self.geneNames = geneNames
        self.exactNetworksIndices = exactNetworksIndices
        self.maxRegs = maxRegs
        self.avgInDegs = self.getAvgInDegs()
        self.avgOutDegs = self.getAvgOutDegs()
        self.triC = self.getTriadicTendencies()
        self.regPotentials = self.getRegPotentials()

    def getAvgInDegs(self):
        avgInDegs = defaultdict(int)
        for refNet in self.referenceNets:
            inDegs = refNet.getInDegs()
            for key, value in inDegs.items():
                avgInDegs[key] += value
        return avgInDegs

    def getAvgOutDegs(self):
        avgOutDegs = defaultdict(int)
        for refNet in self.referenceNets:
            outDegs = refNet.getOutDegs()
            for key, value in outDegs.items():
                avgOutDegs[key] += value
        return avgOutDegs

    def getTriadicTendencies(self):
        triC = np.zeros((self.nNodes, self.nNodes))
        for refNet in self.referenceNets:
            for i in range(self.nNodes):
                for j in range(self.nNodes):
                    if i == j:
                        continue
                    i_regs = [item[0] for item in refNet.getEdges() if item[1] == i]
                    j_regs = [item[0] for item in refNet.getEdges() if item[1] == j]
                    shared_regs = set(i_regs).intersection(set(j_regs))
                    triC[i, j] += len(shared_regs)
        sumTriC = np.sum(triC)
        if sumTriC > 0:
            triC = triC / sumTriC
        return triC

    def getRegPotentials(self):
        return self.regWeights.getRegPotentials()

    def getObj1(self, individual):
        rankedList = []
        nonRankedList = []
        for i in range(len(individual)):
            if individual[i] == 1:
                rankedList.append(self.regPotentials[i])
            else:
                nonRankedList.append(self.regPotentials[i])
        sumN = len(rankedList)
        sumK = len(nonRankedList)
        if sumN == 0 or sumK == 0:
            return 0
        obj1 = sum(rankedList) / sumN - sum(nonRankedList) / sumK
        return obj1

    def getObj2(self, individual):
        obj2 = 0
        indices = np.where(np.array(individual) == 1)[0]
        for i in indices:
            for j in indices:
                if i == j:
                    continue
                obj2 += self.triC[i, j]
        return obj2


class ContextSpecificDecoder:
    def __init__(self, timeSeriesPath, referenceNetPaths, binarisedPath, obj2Weights=None, initialPop=None,
                 initialPopProb=None, exactNetworksIndices=[], decimal="."):
        self.timeSeriesPath = timeSeriesPath
        self.referenceNetPaths = referenceNetPaths
        self.binarisedPath = binarisedPath
        self.obj2Weights = obj2Weights
        self.initialPop = initialPop
        self.initialPopProb = initialPopProb
        self.exactNetworksIndices = exactNetworksIndices
        self.decimal = decimal
        self.binarisedData = BinarisedData(self.binarisedPath, decimal=self.decimal)
        self.geneNames = self.binarisedData.getGeneNames()
        self.geneIndices = self.binarisedData.getGeneIndices()
        self.geneSynonyms = self.binarisedData.getGeneSynonyms()
        self.nNodes = len(self.geneNames)
        self.maxRegs = 3
        method_args = {'regulators': 'all', 'tree_method': 'RF', 'K': 'sqrt', 'ntrees': 1000}
        self.regWeights = self.getRegulatoryWeights(**method_args)
        self.referenceNets = self.getReferenceNetworks(self.referenceNetPaths, self.maxRegs, self.geneIndices,
                                                       self.geneNames, exactNetworksIndices=self.exactNetworksIndices)
        self.netProperties = NetworkProperties(self.referenceNets, self.regWeights, self.nNodes, self.geneIndices,
                                               self.geneNames, exactNetworksIndices=self.exactNetworksIndices,
                                               maxRegs=self.maxRegs)
        self.genSolver = GeneticAlgorithm(self.nNodes, self.maxRegs, self.netProperties, initialPop=self.initialPop,
                                          initialPopProb=self.initialPopProb)

    def getRegulatoryWeights(self, **method_args):
        return self.run_dynGENIE3(**method_args)

    def getReferenceNetworks(self, filePaths, maxRegs, geneIndices, geneNames, exactNetworksIndices=[]):
        referenceNets = []
        for ind, filePath in enumerate(filePaths):
            try:
                df = pd.read_csv(filePath, sep=",", header=None)
                if ind in exactNetworksIndices:
                    df.replace(geneNames, geneIndices, inplace=True)
                else:
                    df.replace(self.geneSynonyms, inplace=True)
                net_object = ReferenceNetwork(df)
                referenceNets.append(net_object)
            except Exception as e:
                print(f"Error while processing {filePath}: {e}")
        return referenceNets

    def run_dynGENIE3(self, **method_args):
        from dynGENIE3 import dynGENIE3
        TS_data = [self.binarisedData.df.values]
        time_points = [np.arange(self.binarisedData.df.shape[0])]
        regulatory_probs, _, _, _, _ = dynGENIE3(TS_data, time_points, **method_args)
        return RegulatoryWeights(regulatory_probs, self.geneNames)

    def getNetworkCandidates(self):
        return self.genSolver.run()

    def run(self):
        """
                Runs the genetic algorithm and returns the entire final population,
                which represents the Pareto front of optimal network solutions.
                """
        # This function call executes the NSGA-II algorithm
        final_population = self.getNetworkCandidates()

        # --- NEW PROGRESS CHECK ---
        print(f"\nGenetic algorithm finished. Found a Pareto front with {len(final_population)} solutions.")

        # We return the entire collection of solutions.
        # Each 'solution' is an 'individual' (a list of 0s and 1s representing a network graph).
        return final_population

    # =========================================================================
    # START: NEW REPLACEMENT FUNCTIONS FOR BOOLEAN INFERENCE
    # =========================================================================

    def format_boolean_function(self, simplified_implicants, regulator_names, target_name):
        if not simplified_implicants or simplified_implicants == {'0'}:
            return f"{target_name}* = 0"
        if simplified_implicants == {'1'}:
            return f"{target_name}* = 1"
        or_terms = []
        for term in simplified_implicants:
            and_terms = []
            str_regulator_names = [str(n) for n in regulator_names]
            for i, bit in enumerate(term):
                if bit == '1':
                    and_terms.append(str_regulator_names[i])
                elif bit == '0':
                    and_terms.append(f"~{str_regulator_names[i]}")
            or_terms.append(" & ".join(and_terms))
        final_expression = " | ".join(or_terms)
        return f"{target_name}* = {final_expression}"

    def getBooleanModel(self, individual):
        from qm import QuineMcCluskey

        # --- PERFORMANCE FIX: Set a maximum number of regulators for which we will attempt full Boolean inference. ---
        # Above this number, the Quine-McCluskey algorithm becomes too slow.
        MAX_REGULATORS_FOR_QM = 12

        boolean_functions = []

        for target_node_idx in range(self.nNodes):
            if individual[target_node_idx] == 1:
                target_node_name = self.geneNames[target_node_idx]
                print(f"  -> Processing target node: {target_node_name}...")

                regulator_indices = self.regWeights.getRegulators(target_node_name)

                # --- DIAGNOSTIC PRINT: Show how many regulators were found. ---
                num_regs = len(regulator_indices)
                print(f"     - Found {num_regs} potential regulators.")

                # --- PERFORMANCE FIX IMPLEMENTATION ---
                if num_regs > MAX_REGULATORS_FOR_QM:
                    print(
                        f"     - WARNING: Too many regulators ({num_regs}). Skipping complex Boolean inference for performance.")
                    # Create a simple placeholder rule based on the gene's average activity.
                    mean_val = self.binarisedData.df[target_node_name].mean()
                    boolean_functions.append(f"{target_node_name}* = {1 if mean_val > 0.5 else 0}")
                    continue  # Move to the next gene

                if num_regs == 0:
                    mean_val = self.binarisedData.df[target_node_name].mean()
                    boolean_functions.append(f"{target_node_name}* = {1 if mean_val > 0.5 else 0}")
                    continue

                # If the number of regulators is acceptable, proceed with the original logic.
                print(f"     - Inferring Boolean function using Quine-McCluskey...")
                regulator_names = [self.geneNames[i] for i in regulator_indices]
                regulator_data = self.binarisedData.df[regulator_names].iloc[:-1]
                target_data = self.binarisedData.df[target_node_name].iloc[1:]
                regulator_data.index = target_data.index
                full_truth_table = pd.concat([regulator_data, target_data], axis=1)
                summary = full_truth_table.groupby(regulator_names)[target_node_name].mean().reset_index()
                minterm_df = summary[summary[target_node_name] > 0.5]
                minterms = []
                if not minterm_df.empty:
                    minterm_inputs = minterm_df[regulator_names]
                    minterms = minterm_inputs.apply(lambda row: int("".join(row.astype(int).astype(str)), 2),
                                                    axis=1).tolist()

                all_possible_inputs = set(range(2 ** num_regs))
                if not summary.empty:
                    observed_inputs_df = summary[regulator_names]
                    observed_inputs = observed_inputs_df.apply(lambda row: int("".join(row.astype(int).astype(str)), 2),
                                                               axis=1).tolist()
                    dont_cares = list(all_possible_inputs - set(observed_inputs))
                else:
                    dont_cares = list(all_possible_inputs)

                if not minterms:
                    boolean_functions.append(f"{target_node_name}* = 0")
                else:
                    qm = QuineMcCluskey()
                    simplified_implicants = qm.simplify(ones=minterms, dc=dont_cares)
                    final_function_str = self.format_boolean_function(simplified_implicants, regulator_names,
                                                                      target_node_name)
                    boolean_functions.append(final_function_str)

        return boolean_functions

    # =========================================================================
    # END: NEW REPLACEMENT FUNCTIONS
    # =========================================================================


class BinarisedData:
    def __init__(self, filePath, decimal="."):
        self.filePath = filePath
        self.decimal = decimal
        self.df = self.read_data()
        self.geneNames = self.getGeneNames()
        self.geneIndices = self.getGeneIndices()
        self.geneSynonyms = self.getGeneSynonyms()

    def read_data(self):
        try:
            df = pd.read_csv(
                self.filePath,
                sep=",",
                header=None,
                dtype=np.float64
            )
            df.columns = [f'x{i}' for i in range(df.shape[1])]
        except Exception as e:
            print(f"CRITICAL ERROR: Could not read and parse the binarised data file: {self.filePath}")
            print(f"Pandas error: {e}")
            print("This may be due to an unexpected format in the CSV file.")
            sys.exit(1)
        return df

    def getGeneNames(self):
        return self.df.columns.tolist()

    def getGeneIndices(self):
        return list(range(len(self.df.columns)))

    def getGeneSynonyms(self):
        synonyms = {}
        for i, name in enumerate(self.geneNames):
            synonyms[name] = i
        return synonyms


def calculate_cod_for_rule(function_str, regulator_names, target_name, binarised_df):
    """
    Calculates the Coefficient of Determination (COD) for a given Boolean rule.
    COD = 1 - (SSE / SST)
    """
    if not regulator_names:
        # If there are no regulators, the prediction is constant.
        # The error is based on the target data's variance.
        # A constant predictor has a COD of 0.
        return 0.0

    # 1. Get the actual target data time series
    actual_target_data = binarised_df[target_name].iloc[1:].values

    # 2. Get the regulator data from the previous time step
    regulator_data = binarised_df[regulator_names].iloc[:-1]

    # 3. Generate predictions for each time step based on the rule
    predicted_target_data = []
    for index, row in regulator_data.iterrows():
        # Create the input state for this time step
        input_state = {str(reg): val for reg, val in row.items()}

        # Evaluate the function string
        py_expr = function_str.replace('&', 'and').replace('|', 'or').replace('~', 'not ')
        for var, val in input_state.items():
            py_expr = re.sub(r'\b' + str(var) + r'\b', str(val), py_expr)

        try:
            prediction = int(eval(py_expr))
            predicted_target_data.append(prediction)
        except Exception:
            # If evaluation fails, it's a bad prediction
            predicted_target_data.append(0)

    predicted_target_data = np.array(predicted_target_data)

    # 4. Calculate Sum of Squared Errors (SSE)
    errors = actual_target_data - predicted_target_data
    sse = np.sum(errors ** 2)

    # 5. Calculate Total Sum of Squares (SST)
    mean_actual_data = np.mean(actual_target_data)
    sst = np.sum((actual_target_data - mean_actual_data) ** 2)

    if sst == 0:
        # If SST is 0, the actual data never changes.
        # If SSE is also 0, the prediction is perfect (COD=1), otherwise it's undefined (return 0).
        return 1.0 if sse == 0 else 0.0

    cod = 1 - (sse / sst)
    return cod





# Place this function inside SAILoR_utils.py

def infer_boolean_function(regulator_names, target_name, binarised_df):
    """
    Infers the single best Boolean function for a given set of regulators.
    This contains the logic previously inside the getBooleanModel method.
    """
    from qm import QuineMcCluskey  # Keep the import local

    num_regs = len(regulator_names)

    # Handle edge cases
    if num_regs == 0:
        mean_val = binarised_df[target_name].mean()
        return "1" if mean_val > 0.5 else "0"

    # This limit is important for performance
    MAX_REGULATORS_FOR_QM = 12
    if num_regs > MAX_REGULATORS_FOR_QM:
        print(f"      - WARNING: Too many regulators ({num_regs}) for Quine-McCluskey. Returning simple rule.")
        mean_val = binarised_df[target_name].mean()
        return "1" if mean_val > 0.5 else "0"

    # Prepare data for Quine-McCluskey
    regulator_data = binarised_df[regulator_names].iloc[:-1]
    target_data = binarised_df[target_name].iloc[1:]
    regulator_data.index = target_data.index
    full_truth_table = pd.concat([regulator_data, target_data], axis=1)
    summary = full_truth_table.groupby(regulator_names)[target_name].mean().reset_index()

    minterm_df = summary[summary[target_name] > 0.5]
    minterms = []
    if not minterm_df.empty:
        minterm_inputs = minterm_df[regulator_names]
        minterms = minterm_inputs.apply(lambda row: int("".join(row.astype(int).astype(str)), 2), axis=1).tolist()

    all_possible_inputs = set(range(2 ** num_regs))
    if not summary.empty:
        observed_inputs_df = summary[regulator_names]
        observed_inputs = observed_inputs_df.apply(lambda row: int("".join(row.astype(int).astype(str)), 2),
                                                   axis=1).tolist()
        dont_cares = list(all_possible_inputs - set(observed_inputs))
    else:
        dont_cares = list(all_possible_inputs)

    if not minterms:
        return "0"
    else:
        qm = QuineMcCluskey()
        simplified_implicants = qm.simplify(ones=minterms, dc=dont_cares)

        # Format the final function string
        if not simplified_implicants or simplified_implicants == {'0'}: return "0"
        if simplified_implicants == {'1'}: return "1"
        or_terms = []
        for term in simplified_implicants:
            and_terms = []
            str_regulator_names = [str(n) for n in regulator_names]
            for i, bit in enumerate(term):
                if bit == '1':
                    and_terms.append(str_regulator_names[i])
                elif bit == '0':
                    and_terms.append(f"~{str_regulator_names[i]}")
            or_terms.append(" & ".join(and_terms))
        return " | ".join(or_terms)

class RegulatoryWeights:
    def __init__(self, reg_probs, gene_names):
        self.reg_probs = reg_probs
        self.geneNames = gene_names
        self.nNodes = self.reg_probs.shape[0]

    def getRegulators(self, node_name):
        # --- NEW: Set a limit on the number of top regulators to consider for each gene. ---
        # This is the most effective way to control computational complexity.
        TOP_N_REGULATORS = 10

        node_idx = self.geneNames.index(node_name)

        # Get the importance scores for all potential regulators of the target node
        regulator_scores = self.reg_probs[:, node_idx]

        # Get the indices of all regulators with a score > 0
        non_zero_indices = np.where(regulator_scores > 0)[0]

        # If there are fewer regulators than our limit, take them all
        if len(non_zero_indices) <= TOP_N_REGULATORS:
            return non_zero_indices
        else:
            # Otherwise, find the indices of the top N regulators by sorting their scores
            # We use `argpartition` for efficiency as it's faster than a full `argsort`
            # The negative sign is to sort in descending order (highest score first)
            partitioned_indices = np.argpartition(-regulator_scores, TOP_N_REGULATORS)[:TOP_N_REGULATORS]
            return partitioned_indices

    def getRegPotentials(self):
        return np.sum(self.reg_probs, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeSeriesPath")
    parser.add_argument("--binarisedPath")
    parser.add_argument("--referencePaths")
    parser.add_argument("--outputFilePath")
    parser.add_argument("--decimal")
    parser.add_argument("--exactNetworksIndices")
    parser.add_argument("--initialPop")
    parser.add_argument("--initialPopProb")
    args = parser.parse_args()
    if args.timeSeriesPath is None or args.binarisedPath is None or args.referencePaths is None:
        sys.exit("Error: Invalid arguments!")
    decimal = args.decimal
    if decimal is None:
        decimal = "."
    exactNetworksIndices = args.exactNetworksIndices
    if exactNetworksIndices:
        exactNetworksIndices = ast.literal_eval(exactNetworksIndices)
    else:
        exactNetworksIndices = []
    initialPop = args.initialPop
    if initialPop:
        initialPop = ast.literal_eval(initialPop)
    initialPopProb = args.initialPopProb
    if initialPopProb:
        initialPopProb = ast.literal_eval(initialPopProb)
    referencePaths = args.referencePaths
    if referencePaths:
        referencePaths = ast.literal_eval(referencePaths)
    else:
        referencePaths = []
    decoder = ContextSpecificDecoder(args.timeSeriesPath, referenceNetPaths=referencePaths,
                                     binarisedPath=args.binarisedPath, obj2Weights=None, initialPop=initialPop,
                                     initialPopProb=initialPopProb, exactNetworksIndices=exactNetworksIndices,
                                     decimal=decimal)

    # --- START: CORRECTED FINAL BLOCK ---
    list_of_models = decoder.run()
    if not list_of_models or not list_of_models[0]:
        print("SAILoR finished but did not produce a valid Boolean model.")
        boolean_expressions = []
    else:
        boolean_expressions = list_of_models[0]
    if args.outputFilePath is None:
        print("Output file path not provided! Printing to standard output.")
        for expression in boolean_expressions:
            print(expression)
    else:
        print(f"\nSaving the best Boolean model to: {args.outputFilePath}")
        with open(args.outputFilePath, "w+") as outFile:
            outFile.writelines([str(expression) + "\n" for expression in boolean_expressions])
    # --- END: CORRECTED FINAL BLOCK ---


if __name__ == "__main__":
    main()