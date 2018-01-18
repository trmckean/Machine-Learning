import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by TylerMcKean on 1/25/16.
 * This class contains functionality to parse the data from the given data file and construct useful Lists of the
 * parsed data to be used by the rest of the program. It has the funcionality to shuffle and standardize the data
 * so that the training algorithm is more correct.
 */
public class InputParser {
    protected String filename;
    private FileInputStream file;
    private BufferedReader buffer;
    private String dataDelim = ",";
    protected List<String[]> inputData = new ArrayList<>();
    protected List<Double[]> integerData = new ArrayList<>();
    protected List<String[]> trainingData = new ArrayList<>();
    protected List<Double[]> trainingIntData = new ArrayList<>();
    protected List<String[]> testData = new ArrayList<>();
    protected List<Double[]> testIntData = new ArrayList<>();

    //Constructor that takes a fileName to set so the parser can read the data
    public InputParser(String filename) {
        this.filename = filename;
    }

    //This function parses the input data and stores it in useful data structures to be used by the rest of the program
    public void parseInput() throws IOException {
        List<String[]> tempData = new ArrayList<>();
        List<Double[]> tempIntData = new ArrayList<>();

        try {
            file = new FileInputStream(filename);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        buffer = new BufferedReader(new InputStreamReader(file));

        for(int i = 0; i < 20000; ++i)
        {
            String inputLine = buffer.readLine();
            String [] tempLine = inputLine.split(dataDelim);
            Double [] tempIntLine = new Double[tempLine.length - 1];
            for(int n = 1; n < tempLine.length; ++n)
            {
               tempIntLine[n-1]  = Double.parseDouble(tempLine[n]);
            }
            tempIntData.add(tempIntLine);
            tempData.add(tempLine);
        }

        this.integerData = tempIntData;
        this.inputData = tempData;

        //Create a training data set
        for(int i = 0; i < 10000; ++i) {
            trainingData.add(tempData.get(i));
            trainingIntData.add(tempIntData.get(i));
        }
        //Create a test data set
        for(int i = 10000; i < 20000; ++i) {
            testData.add(tempData.get(i));
            testIntData.add(tempIntData.get(i));
        }
    }

    //This function simply prints the data that was read in
    public void printData() {
        for(int i = 0; i < inputData.size(); ++i)
        {
            System.out.printf(i + " ");
            for(int n = 0; n < inputData.get(i).length; ++n)
            {
                System.out.printf(inputData.get(i)[n] + " | ");
            }
            System.out.println();
        }
    }

    //This function prints the data structure that only contains numerical values
    public void printIntData() {
        for(int i = 0; i < integerData.size(); ++i)
        {
            System.out.printf(i + " ");
            for(int n = 0; n < integerData.get(i).length; ++n)
            {
                System.out.printf(Double.toString(integerData.get(i)[n]) + " ");
            }
            System.out.println();
        }
    }

    //This function shuffles the training data and then creates a corresponding data structure containing only
    //the numeric values to be used by the program
    public void shuffle() {
            Collections.shuffle(trainingData);
            for(int i = 0; i < 10000; ++i) {
                String [] tempLine = new String[trainingData.get(i).length];
                Double [] tempIntLine = new Double[tempLine.length-1];
                tempLine = trainingData.get(i);
                for(int n = 1; n < tempLine.length; ++n)
                {
                    tempIntLine[n-1] = Double.parseDouble(tempLine[n]);
                }
                trainingIntData.add(i, tempIntLine);
            }
        }

    //This function standardizes the data. It goes through each
    //input column and calculates the mean and standard deviation and then takes the old input and subtracts the mean
    //and divides by the standard deviation. Mean and standard deviation are only calculated using the training data
    //but both the training data and test data are standardized, as per instruction.
    public void standardize() {
        //For each column in the input data
        for(int n = 0; n < 16; ++n) {

            double meanSum = 0.0;
            double mean = 0.0;
            double varianceSum = 0.0;
            double tempVarianceTerm = 0.0;
            double variance = 0.0;
            double standardDeviation = 0.0;


            for (int i = 0; i < trainingIntData.size(); ++i) {
                meanSum += trainingIntData.get(i)[n];
            }
            mean = (meanSum / trainingIntData.size());

            for (int i = 0; i < trainingIntData.size(); ++i) {
                tempVarianceTerm = (trainingIntData.get(i)[n] - mean);
                tempVarianceTerm = (tempVarianceTerm * tempVarianceTerm);
                varianceSum += tempVarianceTerm;
            }

            variance = (varianceSum / (trainingIntData.size() - 1));

            standardDeviation = Math.sqrt(variance);

            //Standardize the training data
            for (int i = 0; i < trainingIntData.size(); ++i) {
                trainingData.get(i)[n+1] = Double.toString((trainingIntData.get(i)[n] - mean)/ standardDeviation);
            }
            //Standardize the test data
            for(int i = 0; i < testIntData.size(); ++i) {

                testIntData.get(i)[n] = ((testIntData.get(i)[n] - mean) / standardDeviation);
            }
        }
    }
}
