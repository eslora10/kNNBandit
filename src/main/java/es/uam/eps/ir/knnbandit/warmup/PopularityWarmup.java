package es.uam.eps.ir.knnbandit.warmup;

import java.util.List;

import org.jooq.lambda.tuple.Tuple2;

import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.basic.PopularityRecommender;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

public class PopularityWarmup<U, I> extends Warmup<U, I> {

	public PopularityWarmup(int nRels, SimpleFastPreferenceData<U, I> prefData, FastUpdateableUserIndex<U> uIndex,
			FastUpdateableItemIndex<I> iIndex) {
		super(nRels, prefData, uIndex, iIndex);
		rec = new PopularityRecommender<U, I>(uIndex, iIndex, prefData, true, threshold);

	}

	@Override
	public List<Tuple2<Integer, Integer>> perform(int iterations, boolean useRatings) {
		loop = new RecommendationLoop<U, I>(uIndex, iIndex, prefData, rec, localMetrics, iterations, useRatings);
        while (!loop.hasEnded()) {
            Double rating = nextIteration();
            if (rating == null) break;
       }
        
        return trainData;
        	
    }

	@Override
	public List<Tuple2<Integer, Integer>> perform(double ratio, boolean useRatings) {
		int nIterations = (int) Math.ceil(ratio*nRels);
		loop = new RecommendationLoop<U, I>(uIndex, iIndex, prefData, rec, localMetrics, 0, useRatings);
		while (!loop.hasEnded() && nIterations > 0) {
            Double rating = nextIteration();
            if (rating == null) break;
            if (rating > 0.0) nIterations--;
		}
		
		return trainData;
	}

}
