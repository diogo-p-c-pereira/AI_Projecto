package algorithms;

import breakout.Breakout;
import utils.Commons;
import utils.IO_Utils;

import java.util.Arrays;

public class BreakOutGA extends GeneticAlghorithm{

    private final static int BREAKOUT_HIDDEN_DIM = 25;

    public BreakOutGA(int seed, int geracoes) {
        super(seed, geracoes);
    }

    @Override
    protected void createPopulation(){
        population = new BreakOutFFNN[N_POPULATION];
        for(int i = 0; i<N_POPULATION; i++){
            population[i] = new BreakOutFFNN(Commons.BREAKOUT_STATE_SIZE, BREAKOUT_HIDDEN_DIM, Commons.BREAKOUT_NUM_ACTIONS);
        }
    }

    @Override
    protected void showGenResult(FeedforwardNeuralNetwork genBest, int i){
        IO_Utils.writeToFile("BreakoutNN_Best_Gen_"+i+".txt",Arrays.toString(genBest.getNeuralNetwork()));
        Breakout b = new Breakout(genBest,SEED);
        System.out.println(genBest.getFitness());
    }

    @Override
    protected FeedforwardNeuralNetwork createNewNN(double[] nn){
        return new BreakOutFFNN(Commons.BREAKOUT_STATE_SIZE, BREAKOUT_HIDDEN_DIM, Commons.BREAKOUT_NUM_ACTIONS, nn);
    }

}
