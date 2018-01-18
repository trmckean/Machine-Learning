import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;

/**
 * Created by TylerMcKean on 1/25/16.
 * This class contains the functionality to actually run the experiments. Here we create an instance of the inputParser
 * class in order to read in the data, then call the standardize function to standardize the data. After that we create
 * an instance of the neuralNetwork and initialize it using parameters representing hidden nodes, learning rate, and
 * the momentum factor. It iterates through a number of epochs and during it each epoch it trains the network on each
 * training example then tests the update network on the training data and the test data and calculates accuracy.
 */
public class NeuralNetworkExperiment {

    public static void main(String[] args) throws IOException {

        InputParser inputParser = new InputParser("letter-recognition.data");
        inputParser.parseInput();
        inputParser.standardize();
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.initialize(4, 0.6, 0.3);
        double trainAccuracy = 0.0;
        int epoch = 0;
        //For loop that can be varied depending on the max number of epochs (n)
        for(int n = 0; n < 100; ++n) {
            //Shuffle the data
            inputParser.shuffle();
            //For loop that trains the network on each training example (10000)
            for (int i = 0; i < 10000; ++i) {
                neuralNetwork.setInputs(inputParser.trainingData.get(i)[0], inputParser.trainingIntData.get(i));
                neuralNetwork.inputToOutput();
            }
            double tempTrainAccuracy = 0;
            //For loop that tests the updated network on each instance of the training data
            for(int i = 0; i < 10000; ++i) {
                neuralNetwork.setInputs(inputParser.trainingData.get(i)[0], inputParser.trainingIntData.get(i));
                tempTrainAccuracy += neuralNetwork.calculateAccuracy();
            }
            double tempTestAccuracy = 0;
            double testAccuracy = 0.0;
            //For loop that tests the updated network on each piece of test data
            for(int i = 0; i < 10000; ++i) {
                neuralNetwork.setInputs(inputParser.testData.get(i)[0], inputParser.testIntData.get(i));
                tempTestAccuracy += neuralNetwork.calculateAccuracy();
            }
            ++epoch;
            //Calculate both the training accuracy and the test accuracy
            trainAccuracy = (tempTrainAccuracy/10000);
            testAccuracy = (tempTestAccuracy/10000);
            System.out.println(epoch + " " + trainAccuracy + " " + testAccuracy);
        }

        System.exit(0);
    }
}
