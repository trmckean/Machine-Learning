package Perceptrons;

import java.io.*;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by TylerMcKean on 1/18/16.
 * This class represents a parser, it is used to grab data from the file and organize it
 */
public class DataParser {
    protected String filename;
    private FileInputStream file;
    private BufferedReader buffer;
    private String dataDelim = ",";
    protected List<String[]> trainingData = new ArrayList<String[]>();
    protected List<String[]> testData = new ArrayList<String[]>();

    public DataParser(String filename) {
        this.filename = filename;
    }

    //This function is used to grab training data from the first 10,000 pieces of data
    public List<String[]> parse(String first, String second) throws IOException {
            file = new FileInputStream("letter-recognition.data");
            buffer = new BufferedReader(new InputStreamReader(file));
            String inputLine = buffer.readLine();
            int i = 0;

            while (inputLine != null && i < 10000) {
                String[] inputs = inputLine.split(dataDelim);

                //System.out.println("First: " + first + " Second: " + second + i);
                ++i;

                if (inputs[0].equals(first) || inputs[0].equals(second)) {
                    //System.out.println("Adding data to training set!" + inputs[0]);
                    trainingData.add(inputs);
                    inputLine = buffer.readLine();
                } else {
                    inputLine = buffer.readLine();
                }
            }
            return trainingData;
        }

    //This function grabs the test data to be used when the main function tests
    public List<String[]> getTestData() throws IOException {
        file = new FileInputStream("letter-recognition.data");
        buffer = new BufferedReader(new InputStreamReader(file));
        String inputLine = buffer.readLine();
        int i = 0;

        while(inputLine != null && i < 10000) {
            inputLine = buffer.readLine();
            ++i;
        }

        while (inputLine != null && i < 20000) {
            String[] inputs = inputLine.split(dataDelim);
            testData.add(inputs);
            inputLine = buffer.readLine();
        }
        return testData;
    }

}
