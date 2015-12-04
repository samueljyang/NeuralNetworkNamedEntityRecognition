package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {

    public static void main(String[] args) throws IOException {
	if (args.length < 2) {
	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
	    return;
	}	    

	// this reads in the train and test datasets
	List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
	List<Datum> testData = FeatureFactory.readTestData(args[1]);	

	//double param = Double.parseDouble(args[2]);
	//int param = Integer.parseInt(args[2]);
	double learningRate = 0.0001;
	double lr = 0.001;
	int H = 100; // size of hidden layer, 100
	int numIterations = 2; // SGD iterations
	
	//String output_file = "../SGD_a="+param+".out";
	String output_file = "../output.out";

	
	//	read the train and test data
	//TODO: Implement this function (just reads in vocab and word vectors)
	FeatureFactory.initializeVocab("../data/vocab.txt");
	SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors.txt"); // n x V
	boolean gradientCheck = false;

	// initialize model 
	// Select between baseline and window models here:
	//BaselineModel model = new BaselineModel();
	WindowModel model;

	if(! gradientCheck) {
	    model = new WindowModel(3, H,lr, learningRate);  // for testing	    
	    //model = new WindowModel(3, 100,lr, learningRate);
	}
	else {
	    model = new WindowModel(3, 2,lr, learningRate); 
	}
	model.initWeights();

	if( gradientCheck) {
	    System.out.println("Checking gradients...");
	    List<SimpleMatrix> Y = model.getY(trainData);
	    List<SimpleMatrix> X = model.getX(trainData);
	    boolean gradientResult;
	    SimpleMatrix input;
	    int errorCount = 0;
	    int numChecks = 10; //Y.size();
	    for (int i =0; i < numChecks; i++ )  { // for each training example, check the gradient
		//   int i = 2; // todo: check all gradients	    
		input = X.get(i);
		gradientResult = GradientCheck.check(Y.get(i), model.weights(input), model.matrixDerivatives(Y.get(i), input), model.objFn()); 
		//System.out.println("GradientCheck: " + gradientResult);
		if(! gradientResult ) {
		    errorCount += 1;
		}
	    }
	    System.out.println("Gradient error on "+errorCount+" of "+numChecks+" training examples.");
	    return;
	}
	//TODO: Implement those two functions
	System.out.println("Training...");
	model.train(trainData, numIterations);
	System.out.println("Testing...");
	model.test(testData, output_file);
	System.out.println("Done");
    }
}