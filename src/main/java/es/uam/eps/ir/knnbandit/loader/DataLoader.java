package es.uam.eps.ir.knnbandit.loader;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parser;
import org.ranksys.formats.parsing.Parsers;

public class DataLoader<U,I>{
	
	Set<U> users = new HashSet<>();
    Set<I> items = new HashSet<>();
    List<Tuple3<U,I,Double>> triplets = new ArrayList<>();
    //List<Tuple2<U, I>> train = new ArrayList<>();
    int numrel = 0;
    //int num_train = 0;
    //int num_test = 0;
    DoubleUnaryOperator weightFunction;
    DoublePredicate relevance;
    
    public DataLoader(boolean useRatings, int threshold) {
    	
    	weightFunction = useRatings ? (double x) -> x :
             (double x) -> (x >= threshold ? 1.0 : 0.0);
        
        relevance = useRatings ? (double x) -> (x >= threshold) : 
        	(double x) -> (x > 0.0);
	    
    }
    
    public void read(String input, Parser<U> parserUser, Parser<I> parserItem) throws FileNotFoundException, IOException {
    	read(input, parserUser, parserItem, "\t");
    }
    
    public void read(String input, Parser<U> parserUser, Parser<I> parserItem, String separator) throws FileNotFoundException, IOException {
    	try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(input))))
	    {
	        String line;
	        while((line = br.readLine()) != null)
	        {
	        	String split[] = line.split(separator);
	            U user = parserUser.parse(split[0]);
	            I item = parserItem.parse(split[1]);
	            double val = Parsers.dp.parse(split[2]);
	            
	            users.add(user);
	            items.add(item);
	            
	            double rating = weightFunction.applyAsDouble(val);
	            /*
	        	if (isTrain()) {
		            num_train++;
		            train.add(new Tuple2<>(user, item)); 
	        		
	        	} else {
	        		num_test++;
	        	}
	        	*/
	            if(relevance.test(rating)) numrel++;
	            
	            triplets.add(new Tuple3<>(user, item, rating));     
	        	
	        }
	    }
    }
    
    public List<Tuple3<U,I,Double>> getRatings(){
    	return this.triplets;
    }
    
    public Set<U> getUsers() {
    	return this.users;
    }
    
    public Set<I> getItems() {
    	return this.items;
    }
    
    public int getRelevants() {
    	return this.numrel;
    }
    
    public boolean isTrain() {
    	return false;
    }
}
