package Perceptrons;


import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by TylerMcKean on 1/17/16.
 * This class represents an individual perceptron, it has multiple data members relevant to training and testing a perceptron
 */
public class Perceptron {
    protected static double learnRate = 0.2;
    protected List<Double> weights = new ArrayList<>();
    protected List<Double> inputs = new ArrayList<>();
    protected String firstLetter;
    protected String secondLetter;
    protected int firstLetterCounter;
    protected int secondLetterCounter;
    protected String expectedValue;
    protected String predictedValue;
    protected double startWeightMin = -1;
    protected double startWeightMax = 1;
    protected DataParser inputParser = new DataParser("letter-recognition.data");
    protected double output;
    protected List<String []> trainingData = new ArrayList<>();
    protected List<String []> testData = new ArrayList<>();
    protected int correct = 0;
    protected int incorrect= 0;
    protected double accuracy;
    protected double lastAccuracy = 0.0;

    //Constructor that defines what perceptron it is, sets initial weights, and grabs the relevant training data
    public Perceptron(String first, String second) throws ParseException, IOException {
        this.firstLetter = first;
        this.secondLetter = second;
        this.setWeights();
        this.setTrainingData();
    }

    public List<Double> getInputs() {
        return inputs;
    }

    public List<Double> getWeights() {
        return weights;
    }

    //Sets the initial weights of the perceptron
    public void setWeights() {
        for(int i = 0; i < 17; ++i) {
            Random randomValue = new Random();
            double startWeight = startWeightMin + (startWeightMax - startWeightMin) * randomValue.nextDouble();
            this.weights.add(i, startWeight);
        }
    }

    public String getName() {
        return (firstLetter + " vs. " + secondLetter);
    }

    public void printWeights() {
        for(int i = 0; i < 17; ++i) {
            System.out.println("Starting weight  w" + i + " " + this.weights.get(i));
        }
    }

    //Sets the inputs for this iteration of the perceptron
    public void setInputs(int n) throws ParseException, IOException {
        this.inputs.add(0, 1.0);
        String [] inputBuffer = trainingData.get(n);
        this.expectedValue = inputBuffer[0];
        for (int i = 1; i < 17; ++i) {
            this.inputs.add(i, (Double.parseDouble(inputBuffer[i]) / 15));
        }
    }

    //Sets the data structure that contains data to be tested on
    public void setTrainingData() throws IOException {
        this.trainingData = inputParser.parse(this.firstLetter, this.secondLetter);

    }

    public void printInputs() {
        System.out.println("Expected Value: " + expectedValue);
        for(int i = 0; i < 17; ++i) {
            System.out.println("Input #" + i + " " + this.inputs.get(i));
        }
    }

    //This function implements the perceptron learning algorithm, it first sets the inputs, checks the output, adjusts the weights, and runs on the next
    public void learn() throws IOException, ParseException {
        correct = 0;
        incorrect = 0;
        for(int n = 0; n < trainingData.size(); ++n) {
            setInputs(n);
            calculateOutput();
            if(predictedValue.equals(expectedValue)) {
                //System.out.println("Predicted the correct value of: " + predictedValue + " for expected value: " + expectedValue);
                if(predictedValue.equals(firstLetter)) {
                    ++firstLetterCounter;
                }
                else {
                    ++secondLetterCounter;
                }
                ++correct;
            }
            else if(predictedValue.equals(firstLetter) && !predictedValue.equals(expectedValue)) {
                for (int i = 0; i < 17; ++i) {
                    weights.set(i, (weights.get(i) + (learnRate * inputs.get(i) * 1)));
                }
                ++firstLetterCounter;
                ++incorrect;
            } else {
                for (int i = 0; i < 17; ++i) {
                    weights.set(i, (weights.get(i) + (learnRate * inputs.get(i) * -1)));
                }
                ++secondLetterCounter;
                ++incorrect;
            }
        }
        if(incorrect == 0)
            accuracy = correct;
        else
            accuracy = correct/incorrect;

        if(accuracy > lastAccuracy) {
            lastAccuracy = accuracy;
            learn();
        }
    }

    //Calculates the summation of the inputs and weights
    public void calculateOutput() {
        output = 0.0;
        for(int i = 0; i < 17; ++i) {
            //System.out.println("Output = "  + output + " weight: " + weights.get(i) + " input: " + inputs.get(i));
            output += (weights.get(i) * inputs.get(i));
        }
        if(output >= 0) {
            predictedValue = secondLetter;
        }
        else {
            predictedValue = firstLetter;
        }
    }

    //This function tests the perceptron on a singular array of data, e.g. a letter and its input values
    public String test(String[] input) {
        this.inputs.add(0, 1.0);
        this.expectedValue = input[0];
        for (int i = 1; i < 17; ++i) {
            this.inputs.add(i, (Double.parseDouble(input[i]) / 15));
        }

        calculateOutput();
        return predictedValue;
    }
}
