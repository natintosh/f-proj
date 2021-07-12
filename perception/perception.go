package main

import (
	"fmt"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/perceptron"
)

func main() {
	rawData, err := base.ParseCSVToInstances("../datasets/blood_datasets_3.csv", true)

	if err != nil {
		fmt.Printf("\n Error occur while parsing data: %s ", err.Error())
		os.Exit(0)
	}

	fmt.Println("******* DATASET ********")
	fmt.Println(rawData)

	//Initialises a new AveragePerceptron classifier
	cls := perceptron.NewAveragePerceptron(3, 1.2, 0.5, 0.3)

	//Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	cls.Fit(trainData)

	predictions := cls.Predict(testData)

	// Prints precision/recall metrics
	confusionMat, _ := evaluation.GetConfusionMatrix(testData, predictions)
	fmt.Println(evaluation.GetSummary(confusionMat))
}
