package main

import (
	"fmt"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
)

func main() {

	var tree base.Classifier

	// Load blood test data for classification
	rawData, err := base.ParseCSVToInstances("../../datasets/blood_datasets_1.csv", true)

	if err != nil {
		fmt.Printf("\n Error occur while parsing data: %s ", err.Error())
		os.Exit(0)
	}

	fmt.Println("******* DATASET ********")
	fmt.Println(rawData)

	shuffledData := base.Shuffle(rawData)

	// Create a 60-40 training-test split
	trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.60)

	tree = ensemble.NewRandomForest(70, 3)
	err = tree.Fit(trainData)
	if err != nil {
		panic(err)
	}

	predictions, err := tree.Predict(testData)
	if err != nil {
		panic(err)
	}

	fmt.Println("RandomForest Performance")

	cf, err := evaluation.GetConfusionMatrix(testData, predictions)

	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}

	fmt.Println(evaluation.GetSummary(cf))
}
