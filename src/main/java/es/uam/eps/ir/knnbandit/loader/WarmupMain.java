package es.uam.eps.ir.knnbandit.loader;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Set;

import org.jooq.lambda.tuple.Tuple2;
import org.jooq.lambda.tuple.Tuple3;
import org.ranksys.formats.parsing.Parsers;

import es.uam.eps.ir.knnbandit.warmup.RandomWarmup;

public class WarmupMain {

	public static void main(String[] args) throws FileNotFoundException, IOException {
		DataLoader<Long, Long> data = new DataLoader<Long, Long>(false, 1);
		data.read("ratings_cm100k.txt", Parsers.lp, Parsers.lp, " ");
		
		List<Tuple3<Long, Long, Double>> ratings = data.getRatings();
		int n = 100;
		
		for(Tuple3<Long, Long, Double> rating : ratings) {
			System.out.println(rating.v1() + " " + rating.v2() + " " + rating.v3());
			if (--n == 0) break;
		}
		
		System.out.println("Num. rels: " + data.getRelevants());
		System.out.println("Total: " + ratings.size());
		
		int nRels = data.getRelevants();
		Set<Long> users = data.getUsers();
		Set<Long> items = data.getItems();
		
		RandomWarmup<Long, Long> warmUp = new RandomWarmup<Long, Long>(ratings, nRels, users, items);
		for (Tuple2<Integer, Integer> train : warmUp.perform(100, false)) {
			System.out.println(train.v1 + "\t" + train.v2);
		}
	}

}
