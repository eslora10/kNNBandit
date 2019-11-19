package es.uam.eps.ir.knnbandit.warmup;

import java.util.List;
import java.util.Optional;
import java.util.Set;

import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;

import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.basic.RandomRecommender;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;

public class RandomWarmup<U, I> extends Warmup<U, I> {

	public RandomWarmup(List<Tuple3<U, I, Double>> ratings, int nRels, Set<U> users, Set<I> items) {
		super(ratings, nRels, users, items);
		rec = new RandomRecommender<U, I>(uIndex, iIndex, prefData, true);
	}

	@Override
	public List<Tuple2<Integer, Integer>> perform(int iterations, boolean useRatings) {
        loop = new RecommendationLoop<U, I>(uIndex, iIndex, prefData, rec, localMetrics, iterations, useRatings);
        while (!loop.hasEnded()) {
            //StringBuilder builder = new StringBuilder();
            //long aa = System.currentTimeMillis();
            Tuple2<Integer, Integer> tuple = loop.nextIteration();
            //long bb = System.currentTimeMillis();
            if (tuple == null){
                break; // The loop has finished
            }
            trainData.add(tuple);
            Optional<IdxPref> value = this.prefData.getPreference(tuple.v1, tuple.v2);
            if(value.isPresent() && value.get().v2 >= threshold)
            	nRels--;
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
        
        return trainData;
        	
    }

	@Override
	public List<Tuple2<Integer, Integer>> perform(double ratio, boolean useRatings) {
		int nIterations = (int) Math.floor(ratio*ratings.size());
		return perform(nIterations, useRatings);
		
	}

}
