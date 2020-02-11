package es.uam.eps.ir.knnbandit.warmup;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.jooq.lambda.tuple.Tuple3;
import org.jooq.lambda.tuple.Tuple2;

import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

public class Warmup<U, I> {
	
	List<Tuple3<U, I, Double>> ratings;
	List<Double> results = new ArrayList<Double>();
	int nRels;
	double threshold = 0.5;
	FastUpdateableUserIndex<U> uIndex;
	FastUpdateableItemIndex<I> iIndex;
	SimpleFastPreferenceData<U, I> prefData;
	InteractiveRecommender<U, I> rec;
	RecommendationLoop<U, I> loop;
    Map<String, CumulativeMetric<U, I>> localMetrics = new HashMap<>();
    List<Tuple2<Integer, Integer>> trainData = new ArrayList<Tuple2<Integer, Integer>>();
    boolean countFails = true;
	
	public Warmup(int nRels, SimpleFastPreferenceData<U, I> prefData, 
			FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, 
			InteractiveRecommender<U, I> rec) {
		this.nRels =  nRels;	
        this.uIndex = uIndex;
        this.iIndex = iIndex;
        this.prefData = prefData;
        this.rec = rec;
        localMetrics.put("recall", new CumulativeRecall<U,I>(prefData, nRels, threshold));
	}
	
	public Warmup(int nRels, SimpleFastPreferenceData<U, I> prefData, 
			FastUpdateableUserIndex<U> uIndex, FastUpdateableItemIndex<I> iIndex, 
			InteractiveRecommender<U, I> rec, boolean countFails) {
		this(nRels,prefData, uIndex, iIndex, rec);
		this.countFails = countFails;
		
	}
	
	public Double nextIteration() {
		double ret; 
		//StringBuilder builder = new StringBuilder();
        //long aa = System.currentTimeMillis();
        Tuple2<Integer, Integer> tuple = loop.nextIteration();
        //long bb = System.currentTimeMillis();
        if (tuple == null){
            return null; // The loop has finished
        }
        results.add(loop.getMetrics().get("recall"));
        Optional<IdxPref> value = this.prefData.getPreference(tuple.v1, tuple.v2);
        /*
        if(value.isPresent() && value.get().v2 >= threshold) {
        	trainData.add(tuple);
        	nRels--;
        	return 1.0;
        }
        	
        
        if (countFails) {
        	trainData.add(tuple);
        }*/
        if(value.isPresent() && value.get().v2 >= threshold) {
        	nRels--;
        	ret = 1.0;
        } else {
        	ret = 0.0;
        }
        
        if (countFails || value.isPresent()) {
        	trainData.add(tuple);
        }
        
        return ret;
        /*
        int iter = loop.getCurrentIteration();
        builder.append(iter);
        builder.append("\t");
        builder.append(tuple.v1);
        builder.append("\t");
        builder.append(tuple.v2);
        Map<String, Double> metricVals = loop.getMetrics();
        for (String name : metricNames){
            builder.append("\t");
            builder.append(metricVals.get(name));
        }
        builder.append("\t");
        builder.append((bb - aa));
        builder.append("\n");
        bw.write(builder.toString());
        */
		
	}
	
	public List<Tuple2<Integer, Integer>> perform(Integer iterations, boolean useRatings) {
		loop = new RecommendationLoop<U, I>(uIndex, iIndex, prefData, rec, localMetrics, iterations, useRatings);
        while (!loop.hasEnded()) {
            Double rating = nextIteration();
            if (rating == null) break;
       }
        
        return trainData;
        	
    }

	public List<Tuple2<Integer, Integer>> perform(Double ratio, boolean useRatings) {
		int nIterations = (int) Math.ceil(ratio*nRels);
		loop = new RecommendationLoop<U, I>(uIndex, iIndex, prefData, rec, localMetrics, 0, useRatings);
		while (!loop.hasEnded() && nIterations > 0) {
            Double rating = nextIteration();
            if (rating == null) break;
            if (rating > 0.0) nIterations--;
		}
		
		return trainData;
		
	}
	
	public int getRelevants() {
		return nRels;
	}
	
	public  Map<String, CumulativeMetric<U, I>> getLocalMetrics(){
		return localMetrics;
	}
}
