package main

import (
	"fmt"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/trees"
)

func main() {
	// Load blood test data for classification
	rawData, err := base.ParseCSVToInstances("../../datasets/blood_datasets_2.csv", true)

	if err != nil {
		fmt.Printf("\n Error occur while parsing data: %s ", err.Error())
		os.Exit(0)
	}

	fmt.Println("******* DATASET ********")
	fmt.Println(rawData)

	criteria := []string{"entropy", "gini"}

	for _, criterion := range criteria {

		fmt.Printf("\n{\n\tCriterion: %s\n}\n", criterion)

		// Shuffle the raw dataset
		shuffledData := base.Shuffle(rawData)

		//Do a training-test split
		trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.60)

		decTree := trees.NewDecisionTreeClassifier(criterion, -1, []int64{0, 1, 2, 3, 4})

		// Train Tree
		err = decTree.Fit(trainData)

		if err != nil {
			panic(err)
		}

		// Print out tree for visualization - shows splits and feature and predictions
		fmt.Println(decTree.String())

		// Access Predictions
		classificationPreds := decTree.Predict(testData)

		fmt.Println("Titanic Predictions")
		fmt.Println(classificationPreds)

		// Evaluate Accuracy on Test Data
		fmt.Println(decTree.Evaluate(testData))
	}
}
