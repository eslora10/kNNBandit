package es.uam.eps.ir.knnbandit.warmup;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.jooq.lambda.tuple.Tuple3;
import org.jooq.lambda.tuple.Tuple2;

import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

public abstract class Warmup<U, I> {
	
	List<Tuple3<U, I, Double>> ratings;
	int nRels;
	double threshold = 0.5;
	FastUpdateableUserIndex<U> uIndex;
	FastUpdateableItemIndex<I> iIndex;
	SimpleFastPreferenceData<U, I> prefData;
	InteractiveRecommender<U, I> rec;
	RecommendationLoop<U, I> loop;
    Map<String, CumulativeMetric<U, I>> localMetrics = new HashMap<>();
    List<Tuple2<Integer, Integer>> trainData = new ArrayList<Tuple2<Integer, Integer>>();
	
	public Warmup(List<Tuple3<U, I, Double>> ratings, int nRels, Set<U> users, Set<I> items) {
		this.ratings = ratings;
		this.nRels =  nRels;	
        uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        iIndex = SimpleFastUpdateableItemIndex.load(items.stream());
        prefData = SimpleFastPreferenceData.load(ratings.stream(), uIndex, iIndex);
        localMetrics.put("Cummulative recall", new CumulativeRecall<U,I>(prefData, nRels, threshold));
	}
	
	public abstract List<Tuple2<Integer, Integer>> perform(int iterations, boolean useRatings);
	
	public abstract List<Tuple2<Integer, Integer>> perform(double ratio, boolean useRatings);

}
