package cs224n.deep;
import java.lang.*;
import java.util.*;
import java.lang.Math.*;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;


import org.ejml.data.*;
import org.ejml.simple.*;



import java.text.*;

public class WindowModel {

        protected SimpleMatrix L, W, Wout, b1, b2;
	//
        public int windowSize,wordSize, hiddenSize, K;
        public double lr, learningRate;

        public WindowModel(int _windowSize, int _hiddenSize, double _lr, double _learningRate){
	    hiddenSize = _hiddenSize;
	    windowSize = _windowSize;
	    wordSize = 50;
	    K = 5; // number of output classes
	    lr = _lr; // regularization
	    learningRate = _learningRate;
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		//TODO
		// initialize with bias inside as the last column
	        // or separately
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	    int C = windowSize; // context window
	    int n = wordSize; // length of word vector
	    int H = hiddenSize; // Hidden layer size
	    boolean useRandomL = false;
	    //int K = 5; // number of output classes (above)
	    
	    double epsilon = Math.sqrt(6.0)/Math.sqrt(n*C + H);
	    System.out.println("Using uniform random initialization with epsilon = " + epsilon);
	    L = FeatureFactory.allVecs; // allVecs (L) is n x V, but data input is n*C x 1
	    if( useRandomL) {
		L = SimpleMatrix.random(L.numRows(), L.numCols(), -epsilon, epsilon, new Random() );
	    }
	    W = SimpleMatrix.random(H,C*n, -epsilon, epsilon, new Random() );
	    Wout = SimpleMatrix.random(K,H, -epsilon, epsilon, new Random()); // U	    
	    b1 = new SimpleMatrix(H,1);
	    b2 = new SimpleMatrix(K,1);
	    System.out.println("Done initializing.");
	}
        public int getIndex(String word) {
	    Object o = FeatureFactory.wordToNum.get(word);
	    int i1 = 0;
	    if( o != null)
		i1 = (int) o;
	    return i1;
	}

        public List<SimpleMatrix> getX(List<Datum> trainData) {
	    List<SimpleMatrix> X = new ArrayList<SimpleMatrix>();
	    SimpleMatrix Xi, before, center;
	    
	    int i1, i2, i3, index;
	    for(int i = 0; i < trainData.size(); i++ ) {
		Xi = new SimpleMatrix(windowSize * wordSize,1);	
		index = i-1;
		if( index < 0)
		    index = 0;
		i1 = getIndex(trainData.get(index).word);		
		i2 = getIndex(trainData.get(i).word);
		index = i+1;
		if (index == trainData.size() )
		    index -= 1; // TODO: integrate start/stop tokens
		i3 = getIndex(trainData.get(index).word);	    
		before = L.extractMatrix(0, L.numRows(), i1, i1+1);
		center = before.combine(before.numRows(), 0, L.extractMatrix(0, L.numRows(), i2, i2+1));
		Xi = center.combine(center.numRows(),0 , L.extractMatrix(0, L.numRows(), i3, i3+1));
		X.add(Xi);
	    }
	    return  X;
	}
        public List<SimpleMatrix> getY(List<Datum> trainData) {
	    List<SimpleMatrix> Y = new ArrayList<SimpleMatrix>();
	    SimpleMatrix Yi;
	    for(int i = 0; i < trainData.size(); i++ ) {
		Yi = new SimpleMatrix(5,1);
		String trueLabel = trainData.get(i).label;
		if (trueLabel.equals("O")  ) {
		    Yi.set(0,0, 1);
		} else 	    if (trueLabel.equals("LOC")  ) {
		    Yi.set(1,0, 1);
		} else 	    if (trueLabel.equals("MISC")  ) {
		    Yi.set(2,0, 1);
		} else	    if (trueLabel.equals("ORG")  ) {
		    Yi.set(3,0, 1);
		} else	    if (trueLabel.equals("PER")  ) {
		    Yi.set(4,0, 1);
		} 
		Y.add(Yi);
	    }
	    return  Y;

	}
    
    public SimpleMatrix derivative_b2(SimpleMatrix Y, SimpleMatrix input) {
	return delta2(Y, input);
    }
    public SimpleMatrix derivative_b1(SimpleMatrix Y, SimpleMatrix input) {
	return delta1(Y, input);
    }
    public SimpleMatrix derivative_Wout(SimpleMatrix Y, SimpleMatrix input) {	
	SimpleMatrix d2 = delta2(Y, input);
	SimpleMatrix a = f(W.mult(input).plus(b1));
	SimpleMatrix regTerm = Wout.scale(lr);
	return outerProduct(d2, a).plus(regTerm);
    }
    public SimpleMatrix outerProduct(SimpleMatrix d2, SimpleMatrix a) {
	if( d2.numCols() != 1 || a.numCols() != 1) {
	    throw new RuntimeException();
	}
	SimpleMatrix result = new SimpleMatrix(d2.numRows(), a.numRows());
	for(int i = 0; i < result.numRows(); i++) {
	    for(int j=0; j<result.numCols(); j++) {
		result.set(i,j, d2.get(i,0) * a.get(j,0));
	    }
	}
	return result;
    }
    public SimpleMatrix derivative_W(SimpleMatrix Y, SimpleMatrix input) {
	SimpleMatrix d1 = delta1(Y, input);
	SimpleMatrix x = input;
	SimpleMatrix regTerm = W.scale(lr);
	return outerProduct(d1, x).plus(regTerm);
    }
    public SimpleMatrix derivative_L(SimpleMatrix Y, SimpleMatrix input) {
	SimpleMatrix d1 = delta1(Y, input);	
	SimpleMatrix result = d1.transpose().mult(W).transpose();
	if (result.numCols() != 1 || result.numRows() !=wordSize*windowSize) {
	    System.out.println("Invalid L derivative size.");
	    throw new RuntimeException() ;
	}
	return result;
    }
    public SimpleMatrix delta2(SimpleMatrix Y, SimpleMatrix input) {
	// d2 = p(xi) - yi
	return feedForwardHypothesis(input).minus(Y);
    }
    public SimpleMatrix delta1(SimpleMatrix Y, SimpleMatrix input) {	
	SimpleMatrix d2 = delta2(Y, input);
	SimpleMatrix fz = f(W.mult(input).plus(b1));
	SimpleMatrix ones = onesVector(fz.numRows());
	SimpleMatrix fzsq = multElement(fz,fz);
	SimpleMatrix result = multElement(d2.transpose().mult(Wout).transpose(), ones.minus(fzsq));
	if ( result.numCols() != 1 && result.numRows() != hiddenSize) {
	    System.out.println("delta1 size is incorrect");
	    throw new RuntimeException();
	}
	return result;
    }
    public SimpleMatrix onesVector(int dim) {
	SimpleMatrix ones = new SimpleMatrix(dim, 1);
	for(int i = 0; i < dim; i++ ) {
	    ones.set(i,0,1);

	}
	return ones;
    }
    public SimpleMatrix multElement(SimpleMatrix A, SimpleMatrix B) {
	if (A.numRows() != B.numRows() || A.numCols() != B.numCols() ) {
	    System.out.println("multElement() error: A is " +A.numRows() +"x"+A.numCols() +", B is " +B.numRows() +"x"+B.numCols());
	    throw new RuntimeException();
	}
	SimpleMatrix result = new SimpleMatrix(A.numRows(), A.numCols() );
	for(int i = 0; i < A.numRows(); i++ ) {
	    for(int j =0; j < A.numCols(); j++ ) {
		result.set(i, j, A.get(i,j)*B.get(i,j));
	    }
	}
	return result;
    }
    public List<SimpleMatrix> weights(SimpleMatrix input) {
	List<SimpleMatrix> weights = new ArrayList<SimpleMatrix>();
	weights.add(b2);
	weights.add(b1);
	weights.add(Wout);
	weights.add(W);
	weights.add(input);
	return weights;
    }

    public List<SimpleMatrix> matrixDerivatives(SimpleMatrix Y, SimpleMatrix input) {
	List<SimpleMatrix> matrixDerivatives = new ArrayList<SimpleMatrix>();
	matrixDerivatives.add(derivative_b2(Y, input));
	matrixDerivatives.add(derivative_b1(Y, input));
	matrixDerivatives.add(derivative_Wout(Y, input));
	matrixDerivatives.add(derivative_W(Y, input));
	matrixDerivatives.add(derivative_L(Y, input));// TODO
	return matrixDerivatives;	
    }
    public class LogLikelihoodObjectiveFunction implements ObjectiveFunction {
	public LogLikelihoodObjectiveFunction() {}
	public double valueAt(SimpleMatrix label, SimpleMatrix input) {
	    double value = 0.0;
	    SimpleMatrix p = feedForwardHypothesis(input);
	    for(int k = 0; k < K; k++) {		
		//J(theta) = -1/m *sum{m}, sum{k} yki * log(pk(xi))
		double pk = p.get(k,0);
		double yk = label.get(k,0);
		value = value + -1*yk*Math.log(pk);
	    }
	    // regularization terms
	    for(int i = 0; i < W.numRows(); i++) {
		for(int j =0; j < W.numCols(); j++) {
		    value += lr/2*W.get(i,j)*W.get(i,j);
		}
	    }
	    for(int i = 0; i < Wout.numRows(); i++) {
		for(int j =0; j < Wout.numCols(); j++) {
		    value += lr/2*Wout.get(i,j)*Wout.get(i,j);
		}
	    }
	    return value;
	}	
    }
    public ObjectiveFunction objFn() {
	ObjectiveFunction objFn = new LogLikelihoodObjectiveFunction();
	return objFn;

    }

    public SimpleMatrix g(SimpleMatrix a) {
	// soft-max, dimension K
	if(a.numRows() != K || a.numCols() != 1) {
	    System.out.println("g() error: expected "+K+" x 1"+ " but a is "+a.numRows()+" x "+a.numCols());
	    throw new RuntimeException();
	}
	SimpleMatrix result = new SimpleMatrix(K,1);
	double denom = 0;
	//System.out.println(a.numCols()+" "+a.numRows() );
	double c = a.elementMaxAbs();
	// Exp will overflow and give nan. Use trick here:
	// http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression#Step_2:_Implement_softmaxCost
	for(int j = 0; j < K; j++) {
	    denom = denom + Math.exp(a.get(j,0) -c);
	}
	for(int i = 0; i < K; i++ ) {
	    result.set(i,0,Math.exp(a.get(i,0) -c)/denom);
	}
	return result;
    }
    public double tanh(double ai) {
	return (Math.exp(ai)-Math.exp(-ai))/(Math.exp(ai)+Math.exp(-ai));
    }
    public SimpleMatrix f(SimpleMatrix z) {
	//tanh, dimension hiddenSize
	if(z.numRows() != hiddenSize || z.numCols() != 1) {
	    System.out.println("f() error: expected "+hiddenSize+" x 1"+ " but z is "+z.numRows()+" x "+z.numCols());
	    throw new RuntimeException();
	}

	SimpleMatrix result = new SimpleMatrix(hiddenSize,1);
	for(int i = 0; i < hiddenSize; i++ ) {
	    double zi = z.get(i,0);
	    result.set(i,0, tanh(zi));
	}
	return result;
    }
	public SimpleMatrix feedForwardHypothesis(SimpleMatrix x) {
	    //System.out.println(W.numRows() +" "+ x.numRows() +" " +b1.numRows());
	    //System.out.println(W.numCols() +" "+ x.numCols() +" " +b1.numCols());
	    
	    SimpleMatrix z = W.mult(x).plus(b1);
	    SimpleMatrix h = f(z);
	    SimpleMatrix a = Wout.mult(h).plus(b2);
	    SimpleMatrix result = g(a);
	    return result;
    }



	/**
	 * Simplest SGD training 
	 */
    public void train(List<Datum> trainData, int numIterations ){	    
	
	    double alpha = learningRate; // learning rate
	    List<SimpleMatrix> Y = getY(trainData);
	    List<SimpleMatrix> X = getX(trainData);
	    SimpleMatrix Xi, Yi, L1, L2, L3, dWout, dW, db1, db2, dL;
	    int iL1, iL2, iL3, index;	    
	    for(int j= 0; j < numIterations; j++) {
		//	TODO		
		for(int i = 1; i < trainData.size()-1; i++ ) {
		    if(i % 1000 == 0)
			System.out.println("Iter " + j + " of "+ numIterations+" i="+i+" of " +trainData.size());
		    Xi = X.get(i);
		    Yi = Y.get(i);
		    dWout = derivative_Wout(Yi, Xi);
		    dW = derivative_W(Yi, Xi);
		    db1 = derivative_b1(Yi, Xi);
		    db2 = derivative_b2(Yi, Xi);
		    dL = derivative_L(Yi, Xi);
		    // update parameters
		    Wout = Wout.minus(dWout.scale(alpha));
		    W = W.minus(dW.scale(alpha));
		    b1 = b1.minus(db1.scale(alpha));
		    b2 = b2.minus(db2.scale(alpha));
		    b2 = b2.minus(db2.scale(alpha));

		    // L update
		    iL1 = getIndex(trainData.get(i-1).word);		
		    iL2 = getIndex(trainData.get(i).word);		
		    iL3 = getIndex(trainData.get(i+1).word);		
		    L1 = L.extractMatrix(0, L.numRows(), iL1, iL1+1).minus(dL.extractMatrix(0, wordSize, 0,1).scale(alpha));
		    L2 = L.extractMatrix(0, L.numRows(), iL2, iL2+1).minus(dL.extractMatrix(wordSize, 2*wordSize, 0,1).scale(alpha));
		    L3 = L.extractMatrix(0, L.numRows(), iL3, iL3+1).minus(dL.extractMatrix(2*wordSize, 3*wordSize, 0,1).scale(alpha));
		    
		    L.insertIntoThis(0, iL1, L1);
		    L.insertIntoThis(0, iL2, L2);
		    L.insertIntoThis(0, iL3, L3);
		}
	    }
		
	}

	
    public void test(List<Datum> testData, String output_file){
	    List<SimpleMatrix> X = getX(testData);
	    String[] classMapping = new String[5];
	    classMapping[0] = "O";
	    classMapping[1] = "LOC";
	    classMapping[2] = "MISC";
	    classMapping[3] = "ORG";
	    classMapping[4] = "PER";

	    File file = new File(output_file);

	    // if file doesnt exists, then create it
	    try {
		if (!file.exists()) {
		    file.createNewFile();
		}

	    FileWriter fw = new FileWriter(file.getAbsoluteFile());
	    BufferedWriter bw = new BufferedWriter(fw);

	    SimpleMatrix Yi;
	    String predictedLabel, testWord, trueLabel;
	    int index;
	    double Yimax;
	    for(int i = 0; i < testData.size(); i++ ) {
		testWord = testData.get(i).word;
		trueLabel = testData.get(i).label;
		Yi = feedForwardHypothesis(X.get(i));
		//System.out.println(Yi);
		// get index from Yi
		index = 0;
		Yimax = Yi.get(0,0);
		for(int j = 0; j < Yi.numRows(); j++) {
		    //System.out.print(Yi.get(j,0)+'\t');
		    if(Yi.get(j,0) > Yimax) {
			index = j;
		    }
		}		
		predictedLabel = classMapping[index];

		//write result to text file
		bw.write(testWord + "\t" + trueLabel + "\t" + predictedLabel + "\n");
	    }
	    bw.close();
	    } catch (Exception e) {}
		// TODO
	}
	
}
