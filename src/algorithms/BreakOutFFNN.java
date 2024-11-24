package algorithms;

import breakout.BreakoutBoard;

public class BreakOutFFNN extends FeedforwardNeuralNetwork{
    public BreakOutFFNN(int inputDim, int hiddenDim, int outputDim) {
        super(inputDim, hiddenDim, outputDim);
    }

    public BreakOutFFNN(int inputDim, int hiddenDim, int outputDim, double[] values) {
        super(inputDim,hiddenDim,outputDim,values);
    }

    @Override
    public void calculateFitness(int seed){
        BreakoutBoard a = new BreakoutBoard(this,false, seed);
        Thread t1 = new Thread(() -> a.runSimulation());

        BreakoutBoard b = new BreakoutBoard(this,false, seed*2);
        Thread t2 = new Thread(() -> b.runSimulation());

        BreakoutBoard c = new BreakoutBoard(this,false, seed*3);
        Thread t3 = new Thread(() -> c.runSimulation());

        startThreads(t1,t2,t3);

        this.fitness=((a.getFitness()+b.getFitness()+c.getFitness())/3);
    }
    @Override
    public int nextMove(int[] currentState){
        return super.nextMove(currentState)+1;
    }

}
