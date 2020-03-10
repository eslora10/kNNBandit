package es.uam.eps.ir.knnbandit.loader;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.FastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableItemIndex;
import es.uam.eps.ir.knnbandit.data.preference.index.fast.SimpleFastUpdateableUserIndex;
import es.uam.eps.ir.knnbandit.metrics.CumulativeMetric;
import es.uam.eps.ir.knnbandit.metrics.CumulativeRecall;
import es.uam.eps.ir.knnbandit.recommendation.InteractiveRecommender;
import es.uam.eps.ir.knnbandit.recommendation.RecommendationLoop;
import es.uam.eps.ir.knnbandit.selector.AlgorithmSelector;
import es.uam.eps.ir.knnbandit.selector.UnconfiguredException;
import es.uam.eps.ir.knnbandit.warmup.*;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;

public class WarmupMain {

	public static void main(String[] args) throws FileNotFoundException, IOException, UnconfiguredException {
		
		String warmupAlg = args[0];
		String exploitAlg = args[1];
		Double p = Double.parseDouble(args[2]);
		boolean countFails = Boolean.parseBoolean(args[3]);;
		
		DataLoader<Long, Long> data = new DataLoader<Long, Long>(false, 1);
		data.read("../ratings_cm100k.txt", Parsers.lp, Parsers.lp, " ");
		
		List<Tuple3<Long, Long, Double>> ratings = data.getRatings();
		
		
		//System.out.println("Num. rels: " + data.getRelevants());
		//System.out.println("Total: " + ratings.size());
		
		int nRels = data.getRelevants();
		Set<Long> users = data.getUsers();
		Set<Long> items = data.getItems();

		FastUpdateableUserIndex<Long> uIndex;
		FastUpdateableItemIndex<Long> iIndex;
		SimpleFastPreferenceData<Long, Long> prefData;
		uIndex = SimpleFastUpdateableUserIndex.load(users.stream());
        iIndex = SimpleFastUpdateableItemIndex.load(items.stream());
        prefData = SimpleFastPreferenceData.load(ratings.stream(), uIndex, iIndex);
		
        AlgorithmSelector<Long, Long> algorithmSelector = new AlgorithmSelector<>();
        algorithmSelector.configure(uIndex, iIndex, prefData, 0.5);
        
		//Beginning of warmup phase
        //double p = 0.3;
        //int p = 100;
        InteractiveRecommender<Long, Long> warmRec = algorithmSelector.getAlgorithm(warmupAlg);
		Warmup<Long, Long> warmUp = new Warmup<Long, Long>(nRels, prefData, uIndex, iIndex, warmRec, countFails);
		List<Tuple2<Integer, Integer>> train;
		if (p < 1) {
			train = warmUp.perform(p, false);
		} else {
			train = warmUp.perform(Integer.parseInt(args[2]), false);
		}
		
		
		//Beginning of exploit phase
		nRels = warmUp.getRelevants();
        
        InteractiveRecommender<Long,Long> rec = algorithmSelector.getAlgorithm(exploitAlg);
        //Training
        rec.update(train);
        //Interactive loop
        Map<String, CumulativeMetric<Long, Long>> metrics = new HashMap<String, CumulativeMetric<Long, Long>>();
        metrics.put("recall", new CumulativeRecall<Long, Long>(prefData, nRels, 0.5));
        RecommendationLoop<Long, Long> loop = new RecommendationLoop<Long, Long>(uIndex, iIndex, prefData, 
        		rec, metrics, 0, false);
        String dir;
        if(countFails) dir = "fails";
        else dir = "noFails";
        
        BufferedWriter writer = new BufferedWriter(new FileWriter(dir+"/"+warmupAlg+"/"+exploitAlg+"/"+p+".txt"));
        
        System.out.println(dir+"/"+warmupAlg+"/"+exploitAlg+"/"+p+".txt");
        while (!loop.hasEnded()) {
        	Tuple2<Integer, Integer> tuple = loop.nextIteration();
            //long bb = System.currentTimeMillis();
            if (tuple == null){
                break; // The loop has finished
            }
            //System.out.println(loop.getMetrics().get("recall"));
            writer.write(loop.getMetrics().get("recall") + "\n");
        }            
        writer.close();

		
		

	}

}
