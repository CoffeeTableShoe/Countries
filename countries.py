import pandas as pd
from dataclasses import dataclass, field
import typing
import random
import logging
import numpy as np

@dataclass
class NodeBaseClass:
    tree: 'typing.Any' = field(default=None, repr=False)
    var: str = None
    cut: float = None
    childLo = None
    childHi = None
    prediction_lo: pd.Series = None
    prediction_hi: pd.Series = None

    def train():
        pass
    
    def evaluate(self, df, y):
        selection_lo = df[df[self.var] <= self.cut]
        selection_hi = df[df[self.var] > self.cut]
        if self.childLo:
            self.childLo.evaluate(selection_lo, y)
        else:
            pd.options.mode.chained_assignment = None
            selection_lo['prediction'] = [self.prediction_lo['mean'] for _ in range(len(selection_lo))]
            selection_lo['prediction_stddev'] = [self.prediction_lo['std'] for _ in range(len(selection_lo))]
            pd.options.mode.chained_assignment = 'warn'
            if self.tree.results is None: 
                self.tree.results = selection_lo
            else:
                self.tree.results = pd.concat([self.tree.results, selection_lo], ignore_index=True)

        if self.childHi:
            self.childHi.evaluate(selection_hi, y)
        else:
            pd.options.mode.chained_assignment = None
            selection_hi['prediction'] = [self.prediction_hi['mean'] for _ in range(len(selection_hi))]
            selection_hi['prediction_stddev'] = [self.prediction_hi['std'] for _ in range(len(selection_hi))]
            pd.options.mode.chained_assignment = 'warn'
            if self.tree.results is None: 
                self.tree.results = selection_hi
            else:
                self.tree.results = pd.concat([self.tree.results, selection_hi], ignore_index=True)


@dataclass
class NodeWithMedianSplitAndGivenVariablesSequence(NodeBaseClass):
    
    def train(self, df, y, vars_left_to_train):
        self.var = vars_left_to_train[0]
        self.cut = df[self.var].median()
        selection_lo = df[df[self.var] <= self.cut]
        selection_hi = df[df[self.var] > self.cut]
        self.prediction_lo = selection_lo[y].describe()
        self.prediction_hi = selection_hi[y].describe()

        if len(vars_left_to_train) > 1:
            self.childLo = NodeWithMedianSplitAndGivenVariablesSequence(self.tree)
            self.childHi = NodeWithMedianSplitAndGivenVariablesSequence(self.tree)
            self.childLo.train(selection_lo, y, vars_left_to_train[1:])
            self.childHi.train(selection_hi, y, vars_left_to_train[1:])

@dataclass
class ClassicRegressionNode(NodeBaseClass):
    
    def getMinSSE(self, df, var, y):
        bestCandadidate = None
        SSEofBestCandidate = None
        values = sorted(df[var])
        for i in range(len(values)-1):
            candidateCut = (values[i] + values[i+1])/2.
            selection_lo = df[df[var] <= candidateCut]
            selection_hi = df[df[var] > candidateCut]
            if len(selection_hi) < self.tree.minDataSplit or len(selection_lo) < self.tree.minDataSplit: continue
            average_lo = selection_lo[y].mean()
            average_hi = selection_hi[y].mean()
            SSE_lo = sum([(yval-average_lo)**2 for yval in selection_lo[y].dropna()])
            SSE_hi = sum([(yval-average_hi)**2 for yval in selection_hi[y].dropna()])
            SSE = SSE_lo + SSE_hi
            if SSE_lo is None or SSE_hi is None:
                print([(yval-average_hi)**2 for yval in selection_hi[y]])
                print([(yval-average_lo)**2 for yval in selection_lo[y]])
                print(average_hi)
                print(average_lo)
                print(SSE)
                print("--------")
            if not SSEofBestCandidate or SSEofBestCandidate > SSE: 
                SSEofBestCandidate = SSE
                bestCandadidate = candidateCut
        if not bestCandadidate :
            logging.warning(f"No candidate found for splitting {self}")
        return (bestCandadidate, SSEofBestCandidate)
            

    def train(self, df, y):
        SSEforBestVar = None
        vars = df.columns
        if self.tree.useRandomVariables :
            numTries = 0
            while(numTries < 100):
                vars = random.sample(vars, self.tree.numRandomVariables)
                if not any([pd.api.types.is_string_dtype(df[v].dtype) for v in vars]): break
                numTries += 1
            if numTries == 100:
                logging.warning(f"Tried 100 times to find appropriate variables for splitting in node {self}. Will stop splitting here and destroy myself")
                return False
        for var in vars:
            if pd.api.types.is_string_dtype(df[var].dtype): continue
            candidateCut, SSEofCandidate = self.getMinSSE(df, var, y)
            if not candidateCut : continue
            if not SSEforBestVar or SSEofCandidate < SSEforBestVar:
                self.var = var
                SSEforBestVar = SSEofCandidate
                self.cut = candidateCut
        
        selection_lo = df[df[self.var] <= self.cut]
        selection_hi = df[df[self.var] >  self.cut]
        self.prediction_lo = selection_lo[y].describe()
        self.prediction_hi = selection_hi[y].describe()
        if len(selection_lo) > 2*self.tree.minDataSplit:
            self.childLo = ClassicRegressionNode(self.tree)
            if not self.childLo.train(selection_lo, y):
                del self.childLo
        if len(selection_hi) > 2*self.tree.minDataSplit:
            self.childHi = ClassicRegressionNode(self.tree)
            if not self.childHi.train(selection_hi, y):
                del self.childHi
        return True


@dataclass
class TreeBaseClass:
    rootNode: NodeBaseClass = None
    results: pd.DataFrame = None

    def evaluate(self, df, y):
        self.rootNode.evaluate(df, y)

@dataclass
class TreeWithMedianSplitAndGivenVariablesSequence(TreeBaseClass):
    vars: list[str] = field(default_factory = list)

    def train(self, df, y):
        self.rootNode = NodeWithMedianSplitAndGivenVariablesSequence(self)
        self.rootNode.train(df, y, vars_left_to_train=self.vars)

@dataclass
class ClassicRegressionTree(TreeBaseClass):
    minDataSplit: int = 5
    useRandomVariables: bool = False
    numRandomVariables: int = 4

    def train(self, df, y):
        self.rootNode = ClassicRegressionNode(self)
        self.rootNode.train(df, y)

def plotResiduals(results, outName):
    
    import matplotlib.pyplot as plt
    results = results.sort_values(by=['prediction'])
    fig, ax = plt.subplots()
    ax.errorbar(range(len(results)), results['prediction'], xerr=0, yerr=results['prediction_stddev'])
    ax.errorbar(range(len(results)), results[y], xerr=0, yerr=0)
    ax.set_ylim(0,120)
    plt.savefig(outName)
    logging.info(f"Sum of residuals: {round(sum(np.abs(results['prediction'] - results[y])),1)}")


if __name__ == "__main__":

    logging.basicConfig(level = logging.INFO)

    loadCols = [
        "Country","Density (P/Km2)","Agricultural Land( %)","Land Area(Km2)",
        "Armed Forces size","Birth Rate","Co2-Emissions","CPI",
        "Fertility Rate","Forested Area (%)","Gasoline Price","GDP",
        "Gross primary education enrollment (%)","Gross tertiary education enrollment (%)","Infant mortality",
        "Life expectancy","Maternal mortality ratio","Minimum wage","Official language",
        "Out of pocket health expenditure","Physicians per thousand","Population","Population: Labor force participation (%)",
        "Total tax rate","Unemployment rate","Urban_population"]
    colNames = [
        "name","density","agriculturalLandRatio","area",
        "armySize","birthRate","Co2Emissions","CPI",
        "fertilityRate","forestedLandRatio","gasPrice","GDP",
        "primaryEducationRatio","tertiaryEducationRatio","infantMortality",
        "lifeExpectancy","maternalMortalityRatio","minWage","language",
        "healthExpenditure","physiciansPerThousands","population","laborForceRatio",
        "taxRate","unemploymentRate","urbanPopulation"]
    
    classificationVars = [
        'density', 'agriculturalLandRatio', 'birthRate', 'CPI', 'fertilityRate', 
        'primaryEducationRatio', 'tertiaryEducationRatio', 'infantMortality', 'physiciansPerThousands'
        'GDP', 'healthExpenditure', 'unemploymentRate']
    
    bestGuessVars = ['fertilityRate', 'tertiaryEducationRatio', 'physiciansPerThousands', 'unemploymentRate']
    


    filePath = "world-data-2023.csv"
    data = pd.read_csv(filePath, quotechar='"', skipinitialspace=True, header=0, usecols=loadCols)
    data.columns = colNames
    data = data.replace('$', '')
    for colName in colNames:
        firstVal = data[colName][0]
        if isinstance(firstVal, str) and '%' in firstVal:
            data[colName] = data[colName].str.rstrip('%').astype('float') / 100.0
        if isinstance(firstVal, str) and ',' in firstVal:
            data[colName] = data[colName].str.replace(',','').str.replace('$','').astype('float')
    data["density"] = data["density"].str.replace(',','').astype('float')

    y = "lifeExpectancy"
    trainingData = data.sample(frac=0.5, random_state=42)
    testingData = data.loc[~data.index.isin(trainingData.index)]

    tree = TreeWithMedianSplitAndGivenVariablesSequence(vars=bestGuessVars)
    tree.train(trainingData, y)
    tree.evaluate(testingData, y)
    print(tree.results.loc[:, [y,'prediction']])

    plotResiduals(tree.results, "test_residuals.pdf")


    tree = ClassicRegressionTree()
    tree.train(trainingData, y)
    tree.evaluate(testingData, y)
    print(tree.results.loc[:, [y,'prediction']])

    plotResiduals(tree.results, "test_classicalTree_residuals.pdf")