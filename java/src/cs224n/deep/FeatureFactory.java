package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
	    // TODO: you'd want to handle sentence boundaries
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
			    data.add(new Datum("</s>","O"));
			    data.add(new Datum("<s>","O"));
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		//TODO implement this
		//set allVecs from filename	
       	        BufferedReader in = new BufferedReader(new FileReader(vecFilename));
		int vocab_count = 0;
		int n = 50;
		int V = wordToNum.size();
		allVecs = new SimpleMatrix(n, V);

       	        for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");			
			if (bits.length != n ) {
			    System.out.println("invalid number of word vector elements");
			    throw new NullPointerException();
			}
			for(int j = 0; j < bits.length; j++) {
			    double value = Double.parseDouble(bits[j]);	
			    //System.out.println("position" + j + " has value " + value);
			    allVecs.set(j, vocab_count, value);
			}
			vocab_count += 1;
			if( vocab_count > V) {
			    System.out.println("Invalid vocab count.");
			    throw new NullPointerException();
			}
		}
		System.out.println("Done reading "+vocab_count+" word vectors and constructed "+n+" x "+V+" matrix."
);
		if (allVecs!=null) return allVecs;
		return null;

	}
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
		//TODO: create this
	    BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
	    int count = 0;
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			wordToNum.put(word, new Integer(count));
			numToWord.put(new Integer(count), word);
			count += 1;
		}
		System.out.println("Done saving vocab.");
		return wordToNum;
	}
 








}
