package main

import (
	"fmt"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	rawData, err := base.ParseCSVToInstances("../datasets/blood_datasets_d0.csv", true)

	if err != nil {
		fmt.Printf("\n Error occur while parsing data: %s ", err.Error())
		os.Exit(0)
	}

	fmt.Println(rawData)

	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	// cls := knn.NewKnnClassifier("euclidean", "kdtree", 3)
	// cls := knn.NewKnnClassifier("manhattan", "linear", 3)
	// cls := knn.NewKnnClassifier("manhattan", "kdtree", 3)
	// cls := knn.NewKnnClassifier("cosine", "linear", 4)
	// cls := knn.NewKnnClassifier("cosine", "kdtree", 4)

	//Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.60)
	cls.Fit(trainData)

	//Calculates the Euclidean distance and returns the most popular label
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
