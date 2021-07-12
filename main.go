package main

import (
	"fmt"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	rawData, err := base.ParseCSVToInstances("datasets/blood_datasets.csv", true)

	if err != nil {
		fmt.Printf("\n Error occur while parsing data: %s ", err.Error())
		os.Exit(0)
	}

	fmt.Println("******* DATASET ********")
	fmt.Println(rawData)

	distFunction := []string{"euclidean", "manhattan", "cosine"}
	algorithm := []string{"linear", "kdtree"}
	kNeighbour := []int{1, 2, 3, 4, 5, 6, 7}

	for _, dist := range distFunction {

		for _, algo := range algorithm {

			for _, k := range kNeighbour {

				fmt.Printf("\n\n\n{ \n\tDistribution Function: %s,\n\tAlgorithm: %s, \n\tkNeighbour: %d \n} \n\n\n", dist, algo, k)

				cls := knn.NewKnnClassifier(dist, algo, k)

				// Shuffle the raw dataset
				shuffledData := base.Shuffle(rawData)

				//Do a training-test split
				trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.60)
				cls.Fit(trainData)

				//Calculates the Euclidean distance and returns the most popular label
				trainingPredictions, err := cls.Predict(trainData)
				if err != nil {
					panic(err)
				}

				testPredictions, err := cls.Predict(testData)
				if err != nil {
					panic(err)
				}

				// Prints precision/recall metrics
				trainConfusionMat, err := evaluation.GetConfusionMatrix(trainData, trainingPredictions)
				if err != nil {
					panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
				}

				fmt.Printf("Summary for evaluation of training data\n")
				fmt.Println(evaluation.GetSummary(trainConfusionMat))

				// Prints precision/recall metrics
				confusionMat, err := evaluation.GetConfusionMatrix(testData, testPredictions)
				if err != nil {
					panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
				}

				fmt.Printf("Summary for evaluation of training data\n")
				fmt.Println(evaluation.GetSummary(confusionMat))

			}
		}
	}
}
