import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by TylerMcKean on 1/25/16.
 * This class represents a neural network. It contains several arrays and lists that contain the current inputs,
 * the weight matrices, the outputs of particular nodes, the error values of particular nodes, as well as parameters
 * such as learning rate, momentum, and the target class of the current inputs.
 */
public class NeuralNetwork {
    protected int hiddenNodes;
    protected double [] inputs = new double[16];
    //Weight matrix to hold weights from the input nodes ( & bias) to the hidden Nodes
    protected List<Double[]> inputToHiddenLayerWeights = new ArrayList<>();
    //Weight matrix to hold weights from the hidden nodes ( & bias) to the output nodes
    protected List<Double[]> hiddenLayerToOutputLayerWeights = new ArrayList<>();
    //Weight matrices to hold the previous change in weights for momentum
    protected List<Double[]> previousInputWeights = new ArrayList<>();
    protected List<Double[]> previousHiddenWeights = new ArrayList<>();
    //Arrays that hold the outputs of each node, first is for hidden nodes, second is for output nodes
    protected double [] hiddenLayerOutput;
    protected double [] outputs = new double[26];
    //Arrays that hold the error for a given node, first is for output nodes, second is for hidden nodes
    protected double [] outputError = new double[26];
    protected double [] hiddenError;
    protected double learningRate;
    protected double momentum;
    protected double startWeightMin = -0.25;
    protected double startWeightMax = 0.25;
    //Integer that represents the target class for a given input
    protected int targetClass;

    //This functiona initializes the neural network based on the given parameters. It takes in an integer
    //representing the number of hidden nodes the network will have as well as a double to assign for the
    //learning rate and the momentum factor.
    protected void initialize(int numHiddenNodes, double learnRate, double momentumRate) {
        hiddenNodes = numHiddenNodes;
        hiddenLayerOutput = new double[hiddenNodes];
        hiddenError = new double[hiddenNodes];
        learningRate = learnRate;
        momentum = momentumRate;
        Double [] temphiddenBias = new Double[hiddenNodes];
        Double [] tempPreviousHiddenBiasWeights = new Double[hiddenNodes];
        //Initializes the starting weights to random values between the given mins and maxes as well as intializes
        //the previous weight matrix to be full of 0s.
        for(int i = 0; i < temphiddenBias.length; ++i)
        {
            Random randomValue = new Random();
            Double startWeight = startWeightMin + (startWeightMax - startWeightMin) * randomValue.nextDouble();
            temphiddenBias[i] = startWeight;
            tempPreviousHiddenBiasWeights[i] = 0.0;
        }
        inputToHiddenLayerWeights.add(0, temphiddenBias);
        previousInputWeights.add(0, tempPreviousHiddenBiasWeights);

        //Initializes the starting weights for the input -> hidden node weight matrix and sets the previous weight
        //matrix (for momentum) to 0s.
        for(int i = 1; i <= hiddenNodes; ++i)
        {
            Double [] tempVector = new Double[16];
            Double [] tempPreviousInputWeight = new Double[16];
            for(int n = 0; n < 16; ++n)
            {
                Random randomValue = new Random();
                Double startWeight = startWeightMin + (startWeightMax - startWeightMin) * randomValue.nextDouble();
                tempVector[n] = startWeight;
                tempPreviousInputWeight[n] = 0.0;
            }
            inputToHiddenLayerWeights.add(i, tempVector);
            previousInputWeights.add(i, tempPreviousInputWeight);
        }
        //Initializes the hidden layer -> output layer weight matrix to random values and sets the previous weight
        //matrix (for momentum) to 0s
        for(int i = 0; i <= hiddenNodes; ++i)
        {
            Double [] tempVector = new Double[26];
            Double [] tempPreviousHiddenWeight = new Double[26];
            for(int n = 0; n < 26; ++n)
            {
                Random randomValue = new Random();
                Double startWeight = startWeightMin + (startWeightMax - startWeightMin) * randomValue.nextDouble();
                tempVector[n] = startWeight;
                tempPreviousHiddenWeight[n] = 0.0;
            }
            hiddenLayerToOutputLayerWeights.add(i, tempVector);
            previousHiddenWeights.add(i, tempPreviousHiddenWeight);
        }
    }

    //This function simply prints the current set of weights for the neural network
    protected void printWeights() {
        for(int i = 0; i < inputToHiddenLayerWeights.size(); ++i) {
            for(int n = 0; n < inputToHiddenLayerWeights.get(i).length; ++n)
            {
                System.out.printf(inputToHiddenLayerWeights.get(i)[n].toString() + " " );
            }
            System.out.println();
        }
        System.out.println("End of Input weights, now hidden layer weights:");
        for(int i = 0; i < hiddenLayerToOutputLayerWeights.size(); ++i) {
            for(int n = 0; n < hiddenLayerToOutputLayerWeights.get(i).length; ++n)
            {
                System.out.printf(hiddenLayerToOutputLayerWeights.get(i)[n].toString() + " " );
            }
            System.out.println();
        }

    }

    //This function contains the algorithm to train the neural network. It starts by using forward propagation
    //from the input layer -> hidden layer -> output layer. It then calculates the error values for each hidden &
    //output node and then applies the backpropagation algorithm to update the weights.
    protected void inputToOutput() {

        //Forward propagation Input -> Hidden Layer
        for(int n = 0; n < hiddenNodes; ++n) {
            double netInput = 0.0;
            double outPut = 0.0;
            for (int i = 0; i < inputs.length; ++i) {
                netInput += (inputToHiddenLayerWeights.get(n+1)[i] * inputs[i]);
            }
            netInput += inputToHiddenLayerWeights.get(0)[n] * 1;
            outPut = (1 / (1 + Math.pow(Math.E, (-1 * netInput))));
            hiddenLayerOutput[n] = outPut;
        }

        //Forward propagation HiddenLayer -> OutputLayer
        for(int n = 0; n < outputs.length; ++n) {
            double outputInput = 0.0;
            double outputOutput = 0.0;
            for (int i = 0; i < hiddenNodes; ++i) {
                outputInput += hiddenLayerOutput[i] * hiddenLayerToOutputLayerWeights.get(i + 1)[n];
            }
                outputInput += hiddenLayerToOutputLayerWeights.get(0)[n] * 1;
                outputOutput = (1 / (1 + Math.pow(Math.E, (-1 * outputInput))));
                outputs[n] = outputOutput;
                outputInput = 0.0;
                outputOutput = 0.0;
            }

        //Determine predicted class
        double max = outputs[0];
        int position = 0;
        for(int i = 0; i < outputs.length; ++i)
        {
            if(outputs[i] > max)
            {
                max = outputs[i];
                position = i;
            }
        }
        position = position + 65;

        //Calculate error value for each output node
        for(int i = 0; i < outputs.length; ++i)
        {
            if(targetClass == (i+65)) {
                outputError[i] = outputs[i] * (1 - outputs[i]) * (0.9 - outputs[i]);
            }
            else {
                outputError[i] = outputs[i] * (1 - outputs[i]) * (0.1 - outputs[i]);
            }
        }

        //Calculate error value for hiddenLayer nodes
        for(int i = 0; i < hiddenError.length; ++i)
        {
            double totalError = 0.0;
            for(int n = 0; n < outputs.length; ++n)
            {
                totalError += hiddenLayerToOutputLayerWeights.get(i+1)[n] * outputError[n];
            }
            hiddenError[i] = hiddenLayerOutput[i] * (1 - hiddenLayerOutput[i]) * (totalError);
        }

        //Backpropagate to change weights for hiddenLayer->outputLayer weights
        for(int i = 0; i < outputs.length; ++i) {
            hiddenLayerToOutputLayerWeights.get(0)[i] = hiddenLayerToOutputLayerWeights.get(0)[i] + ((learningRate * outputError[i] * 1) + momentum * previousHiddenWeights.get(0)[i]);
            previousHiddenWeights.get(0)[i] = (learningRate * outputError[i] * 1);
        }
        for(int n = 1; n <= hiddenNodes; ++n) {
            for (int i = 0; i < outputs.length; ++i) {
                hiddenLayerToOutputLayerWeights.get(n)[i] = hiddenLayerToOutputLayerWeights.get(n)[i] + ((learningRate * outputError[i] * hiddenLayerOutput[n-1]) + momentum * previousHiddenWeights.get(n)[i]);
                previousHiddenWeights.get(n)[i] = (learningRate * outputError[i] * hiddenLayerOutput[n-1]);
            }
        }
        //Backpropagate to change weights for inputLayer->hiddenLayer weights
        for(int i = 0; i < hiddenNodes; ++i) {
            inputToHiddenLayerWeights.get(0)[i] = inputToHiddenLayerWeights.get(0)[i] + ((learningRate * hiddenError[i] * 1) + momentum * previousInputWeights.get(0)[i]);
            previousInputWeights.get(0)[i] = (learningRate * hiddenError[i] * 1);
        }
        for(int n = 1; n <= hiddenNodes; ++n) {
            for (int i = 0; i < inputs.length; ++i) {
                inputToHiddenLayerWeights.get(n)[i] = inputToHiddenLayerWeights.get(n)[i] + ((learningRate * hiddenError[n-1] * inputs[i]) + momentum * previousInputWeights.get(n)[i]);
                previousInputWeights.get(n)[i] = (learningRate * hiddenError[n-1] * inputs[i]);
            }
        }
    }

    //This function returns a 1 or 0 value based on whether the neural network correctly predicted the given target
    //class. It uses forward propagation from the input layer -> hidden layer -> output layer. And then calculates which
    //class was predicted based on which output node had the highest activation.
    protected int calculateAccuracy() {
        //Forward propagation Input -> HiddenLayer
        for(int n = 0; n < hiddenNodes; ++n) {
            double netInput = 0.0;
            double outPut = 0.0;
            for (int i = 0; i < inputs.length; ++i) {
                netInput += (inputToHiddenLayerWeights.get(n+1)[i] * inputs[i]);
            }
            netInput += inputToHiddenLayerWeights.get(0)[n] * 1;
            outPut = (1 / (1 + Math.pow(Math.E, (-1 * netInput))));
            hiddenLayerOutput[n] = outPut;
        }
        //Forward propagation HiddenLayer -> OutputLayer
        for(int n = 0; n < outputs.length; ++n) {
            double outputInput = 0.0;
            double outputOutput = 0.0;
            for (int i = 0; i < hiddenNodes; ++i) {
                outputInput += hiddenLayerOutput[i] * hiddenLayerToOutputLayerWeights.get(i + 1)[n];
            }
            outputInput += hiddenLayerToOutputLayerWeights.get(0)[n] * 1;
            outputOutput = (1 / (1 + Math.pow(Math.E, (-1 * outputInput))));
            outputs[n] = outputOutput;
            outputInput = 0.0;
            outputOutput = 0.0;
        }
        //Determine predicted class
        double max = outputs[0];
        int position = 0;
        for(int i = 0; i < outputs.length; ++i)
        {
            if(outputs[i] > max)
            {
                max = outputs[i];
                position = i;
            }
        }
            position = position + 65;
            if(targetClass == position) {
                return 1;
            }
            else {
                return 0;
            }
        }

    //This function takes in a parameter for the target character for a set of inputs as well as the set of inputs itself.
    //It then sets the inputs of the neural network to the passed in inputs.
    protected void setInputs(String target, Double [] tempInputs) {
        char targetChar = target.charAt(0);
        for(int i = 0; i < tempInputs.length; ++i)
        {
            inputs[i] = tempInputs[i];
        }
        targetClass = (int)targetChar;
    }
}
