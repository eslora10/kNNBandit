package es.uam.eps.ir.knnbandit.loader;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.ranksys.formats.parsing.Parsers;

import org.jooq.lambda.tuple.Tuple2;

import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.metrics.CumulativeGini;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.knnbandit.recommendation.bandits.ItemBanditRecommender;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunction;
import es.uam.eps.ir.knnbandit.recommendation.bandits.functions.ValueFunctions;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.ItemBandit;
import es.uam.eps.ir.knnbandit.recommendation.bandits.item.UCB1ItemBandit;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

public class RandomDataLoader<U, I> extends DataLoader<U, I> {
	Random generator;
	double p;

	public RandomDataLoader(boolean useRatings, int threshold, double p) throws FileNotFoundException, IOException {
		super(useRatings, threshold);
		this.p = p;
		this.generator = new Random();
	}
	
	@Override
	public boolean isTrain() {
		double t = generator.nextDouble();
		return t > (1 - this.p);
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		RandomDataLoader<Integer, Integer> loader = new RandomDataLoader<Integer, Integer>(false, 3, 0.7);
		loader.read("ratings_binary_mini.txt", Parsers.ip, Parsers.ip);
		System.out.println(loader.num_test + " " + loader.num_train);
        FastUpdateableUserIndex<Integer> uIndex = SimpleFastUpdateableUserIndex.load(loader.users.stream());
        FastUpdateableItemIndex<Integer> iIndex = SimpleFastUpdateableItemIndex.load(loader.items.stream());
        SimpleFastPreferenceData<Integer, Integer> prefData = SimpleFastPreferenceData.load(loader.triplets.stream(), uIndex, iIndex);
		InteractiveRecommender<Integer, Integer> recommender = new ItemBanditRecommender<Integer, Integer>(uIndex, iIndex, 
				prefData, true, new UCB1ItemBandit<Integer, Integer>(loader.items.size()), ValueFunctions.identity());
		
		prefData.getUserPreferences(10);
		// Hay que transformar los usuarios e items a uIndex e iIndex
		List<Tuple2<Integer, Integer>> trainTransformed = loader.train.stream().map(
				(t)-> 
				new Tuple2<Integer, Integer>(uIndex.user2uidx(t.v1), 
				iIndex.item2iidx(t.v2))).collect(Collectors.toList());
		
		// Entrenamiento del algoritmo
		recommender.update(trainTransformed);
		
		// Initialize the metrics to compute.
		int numRel = 0;
        Map<String, Supplier<CumulativeMetric<Integer, Integer>>> metrics = new HashMap<>();
        metrics.put("recall", () -> new CumulativeRecall(prefData, numRel, 0.5));
        //metrics.put("gini", () -> new CumulativeGini(loader.items.size()));
        List<String> metricNames = new ArrayList<>(metrics.keySet());
        Map<String, CumulativeMetric<Integer, Integer>> localMetrics = new HashMap<>();
        metricNames.forEach(name -> localMetrics.put(name, metrics.get(name).get()));
		
		// Ejecucion
        RecommendationLoop<Integer, Integer> loop = new RecommendationLoop<>(uIndex, iIndex, 
        		recommender, localMetrics, 0,0);
        
        System.out.println("Iteracion\tUsuario\tItem\tRecall\tTiempo");
        while(!loop.hasEnded())
        {
            StringBuilder builder = new StringBuilder();
            long aa = System.currentTimeMillis();
            Tuple2<Integer, Integer> tuple = loop.nextIteration();
            long bb = System.currentTimeMillis();
            if(tuple == null) break; // The loop has finished
            int iter = loop.getCurrentIteration();
            builder.append(iter);
            builder.append("\t");
            builder.append(tuple.v1);
            builder.append("\t");
            builder.append(tuple.v2);
            Map<String, Double> metricVals = loop.getMetrics();
            for(String name : metricNames)
            {
                builder.append("\t");
                builder.append(metricVals.get(name));
            }
            builder.append("\t");
            builder.append((bb-aa));
            builder.append("\n");
            System.out.println(builder);
        }

		
	}
}
