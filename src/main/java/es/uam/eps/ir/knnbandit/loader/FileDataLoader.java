package es.uam.eps.ir.knnbandit.loader;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.ranksys.formats.parsing.Parser;

public class FileDataLoader<U,I> extends DataLoader<U,I> {
	boolean train;
	
	public FileDataLoader(String input_train, String input_test, boolean useRatings, int threshold) throws FileNotFoundException, IOException  {
		super(useRatings, threshold);
		
	}
	
	@Override
	public void read(String input, Parser<U> parserUser, Parser<I> parserItem) throws FileNotFoundException, IOException {
		this.train = true;
		super.read(input, parserUser, parserItem);
		this.train = false;
		super.read(input, parserUser, parserItem);
		
	}

}
