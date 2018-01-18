//Tyler McKean - Perceptrons
//This program uses the perceptron learning algorithm and the all-pairs classification method in order to predict
//what letter an image represents. This program is comprised of a main method, a perceptron class, and a parser class.

package Perceptrons;

import java.io.IOException;
import java.text.ParseException;
import java.util.*;

//Class that executes the training and testing of the data
public class PerceptronExperiment {

    public static void main(String[] args) throws ParseException, IOException {
        List<Perceptron> perceptrons = new ArrayList<>();
        List<String[]> testData = new ArrayList<String[]>();
        DataParser testParser = new DataParser("letter-recognition.data");
        int totalCorrect = 0;
        int totalIncorrect = 0;
        int totalData = 0;
        String predictedValue = new String();
        Integer[][] confusionMatrix = new Integer[26][26];

        //Sets the confusion matrix values to a default 0 value
        for(int i = 0; i < 26; ++i) {
            for(int j = 0; j < 26; ++j) {
                confusionMatrix[i][j] = 0;
            }
        }

        //Creates a perceptron for every pair of letters
        int p = 0;
        for(char c = 'A'; c <= 'Z'; ++c) {
            for (char x = 'A'; x <= 'Z'; ++x) {
                if (c == x || x < c) {} else {
                        perceptrons.add(new Perceptron(String.valueOf(c), String.valueOf(x)));
                        //System.out.println(perceptrons.get(p).getName());
                        ++p;
                    }
                }
            }

        //System.out.println("Pairs: " + p);
        //This loop runs the learn method for each perceptron using the relevant data, it then reports the accuracy of each perceptron
        for(int i = 0; i < 325; ++i) {
            perceptrons.get(i).learn();
            //System.out.println("#TrainingExamples: " + perceptrons.get(i).trainingData.size() + " correct: " + perceptrons.get(i).correct + " incorrect: " + perceptrons.get(i).incorrect);
            //System.out.println("Predicted First Letter: " + perceptrons.get(i).firstLetterCounter + " Predicted Second Letter: " + perceptrons.get(i).secondLetterCounter);
            totalData += perceptrons.get(i).trainingData.size();
            totalCorrect += perceptrons.get(i).correct;
            totalIncorrect += perceptrons.get(i).incorrect;
        }

        System.out.println("Total Data: " + totalData + " Total Correct: " + totalCorrect + " Total Incorrect: " + totalIncorrect);

        //Grabs the last 10,000 pieces of data in order to test the learned perceptrons on
        testData = testParser.getTestData();

        //This loop runs every perceptron on each input and reports back the letter that received the highest amount of votes
        for(int n = 0; n < testData.size(); ++n) {
            //System.out.println(n);
            HashMap<String, Integer> votes = new HashMap<>();
            //Sets initial value for the hashmap containing the votes for each piece of data
            for(char c = 'A'; c <= 'Z'; ++c) {
                votes.put(String.valueOf(c), 0);
            }
            //Assigns values to the hashmap for each vote the perceptrons return for a given piece of data
            for (int i = 0; i < 325; ++i) {
                String predictedVote = perceptrons.get(i).test(testData.get(n));
                //System.out.println(predictedVote);
                votes.put(predictedVote, votes.get(predictedVote) + 1);
            }
            int maxValue = (Collections.max(votes.values()));
            //Returns the letter with the most votes
            for(Map.Entry<String, Integer> max : votes.entrySet()) {
                if (max.getValue() == maxValue) {
                    predictedValue = max.getKey();
                }
            }
            //System.out.println("predicted: " + predictedValue + "expected: " + testData.get(n)[0]);
            //If the letter it predicted is correct it adjusts the confusion matrix accordingly
            if(predictedValue.equals(testData.get(n)[0])) {
                 int i = 0;
                //Updates the confusion matrix for a correct prediction
                 for(char c = 'A'; c <= 'Z'; ++c) {
                     //System.out.println(i);
                     if(Character.toString(c).equals(predictedValue)) {
                         confusionMatrix[i][i] += 1;
                     }
                     ++i;
                 }
            }
            //If the predicted letter is incorrect it adjusts the confusion matrix accordingly
            else {
                int x = 0;
                int y = 0;
                for (char c = 'A'; c <= 'Z'; ++c) {
                    if (Character.toString(c).equals(testData.get(n)[0])) {
                        for (char m = 'A'; m <= 'Z'; ++m) {
                            if (Character.toString(m).equals(predictedValue)) {
                                confusionMatrix[x][y] += 1;
                            }
                            ++y;
                        }
                    }
                    ++x;
                }
            }
        }

        //Loop to print out the resulting confusion matrix
        for(int x = 0; x < 26; ++x) {
            for(int y = 0; y < 26; ++y) {
                System.out.printf("%d", confusionMatrix[x][y]);
                System.out.print("   |   ");
            }
            System.out.println();
        }

        int correct = 0;
        for(int i = 0; i < 25; ++i)
        {
            correct += confusionMatrix[i][i];
        }

        double accuracy = correct/10000;
        System.out.println("Accuracy: " + accuracy);


        System.exit(0);
    }
}
