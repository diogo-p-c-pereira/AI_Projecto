package algorithms;

import pacman.Pacman;
import utils.Commons;
import utils.IO_Utils;

import java.util.Arrays;

public class PacManGA extends GeneticAlghorithm{

    private final static int PACMAN_HIDDEN_DIM = 100;

    public PacManGA(int seed, int geracoes) {
        super(seed, geracoes);
    }

    @Override
    protected void createPopulation(){
        population = new FeedforwardNeuralNetwork[N_POPULATION];
        for(int i = 0; i<N_POPULATION; i++){
            population[i] = new PacManFFNN(Commons.PACMAN_STATE_SIZE, PACMAN_HIDDEN_DIM, Commons.PACMAN_NUM_ACTIONS);
        }
    }

    @Override
    protected void showGenResult(FeedforwardNeuralNetwork genBest, int i){
        IO_Utils.writeToFile("PacmanNN_Best_Gen_"+i+".txt", Arrays.toString(genBest.getNeuralNetwork()));
        Pacman p = new Pacman(genBest,true ,SEED);
        System.out.println(genBest.getFitness());
    }

    @Override
    protected FeedforwardNeuralNetwork createNewNN(double[] nn){
        return new PacManFFNN(Commons.PACMAN_STATE_SIZE, PACMAN_HIDDEN_DIM, Commons.PACMAN_NUM_ACTIONS, nn);
    }

}
