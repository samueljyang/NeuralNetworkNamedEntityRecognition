package cs224n.deep;
import java.lang.*;
import java.util.*;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class BaselineModel {

	protected SimpleMatrix L, W, Wout;
	public int windowSize,wordSize, hiddenSize;

    List<Datum> trainData;

    public BaselineModel(){
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
	}

	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
	    trainData = _trainData;
	}

	public void test(List<Datum> testData){
	    File file = new File("../baseline.out");

	    // if file doesnt exists, then create it
	    try {
		if (!file.exists()) {
		    file.createNewFile();
		}

	    FileWriter fw = new FileWriter(file.getAbsoluteFile());
	    BufferedWriter bw = new BufferedWriter(fw);

	    for(int i = 0; i < testData.size(); i++ ) {
		String testWord = testData.get(i).word;
		String trueLabel = testData.get(i).label;
		String predictedLabel = "O";
		for(int j =0; j < trainData.size(); j++) {
		    if( testWord.equals(trainData.get(j).word) ) {
			predictedLabel = trainData.get(j).label;
			break;
		    }
		}
		//write result to text file
		bw.write(testWord + "\t" + trueLabel + "\t" + predictedLabel + "\n");
	    }
	    bw.close();
	    } catch (Exception e) {}
	}
	
}
