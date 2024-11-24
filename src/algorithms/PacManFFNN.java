package algorithms;

import pacman.PacmanBoard;

public class PacManFFNN extends FeedforwardNeuralNetwork{
    public PacManFFNN(int inputDim, int hiddenDim, int outputDim) {
        super(inputDim, hiddenDim, outputDim);
    }
    public PacManFFNN(int inputDim, int hiddenDim, int outputDim, double[] values) {
        super(inputDim, hiddenDim, outputDim, values);
    }

    @Override
    public void calculateFitness(int seed){
        PacmanBoard a = new PacmanBoard(this,false, seed);
        Thread t1 = new Thread(() -> a.runSimulation());

        PacmanBoard b = new PacmanBoard(this,false, seed*2);
        Thread t2 = new Thread(() -> b.runSimulation());

        PacmanBoard c = new PacmanBoard(this,false, seed*3);
        Thread t3 = new Thread(() -> c.runSimulation());

        startThreads(t1,t2,t3);

        this.fitness=((a.getFitness()+b.getFitness()+c.getFitness())/3);
    }
}
