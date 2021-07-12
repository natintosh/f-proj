package main

import (
	"fmt"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
)

func main() {
	// Load blood test data for classification
	rawData, err := base.ParseCSVToInstances("../../datasets/blood_datasets_1.csv", true)

	if err != nil {
		fmt.Printf("\n Error occur while parsing data: %s ", err.Error())
		os.Exit(0)
	}

	fmt.Println("******* DATASET ********")
	fmt.Println(rawData)

	var tree base.Classifier

	// Shuffle the raw dataset
	shuffledData := base.Shuffle(rawData)

	// Discretise the blood test dataset with Chi-Merge
	// filt := filters.NewChiMergeFilter(rawData, 0.999)

	// for _, a := range base.NonClassFloatAttributes(rawData) {
	// 	filt.AddAttribute(a)
	// }

	// filt.Train()

	// bloodF := base.NewLazilyFilteredInstances(rawData, filt)

	// Create a 60-40 training-test split
	trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.60)

	//
	// First up, use ID3
	//
	tree = trees.NewID3DecisionTree(0.6)
	// (Parameter controls train-prune split.)

	// Train the ID3 tree
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Generate predictions
	predictions, err := tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate
	fmt.Println("ID3 Performance (information gain)")
	cf, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

	tree = trees.NewID3DecisionTreeFromRule(0.6, new(trees.InformationGainRatioRuleGenerator))
	// (Parameter controls train-prune split.)

	// Train the ID3 tree
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Generate predictions
	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate
	fmt.Println("ID3 Performance (information gain ratio)")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))

	tree = trees.NewID3DecisionTreeFromRule(0.6, new(trees.GiniCoefficientRuleGenerator))
	// (Parameter controls train-prune split.)

	// Train the ID3 tree
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Generate predictions
	predictions, err = tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate
	fmt.Println("ID3 Performance (gini index generator)")
	cf, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(cf))
}
