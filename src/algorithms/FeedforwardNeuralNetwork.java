package algorithms;

import utils.GameController;
import java.util.Random;

public abstract class FeedforwardNeuralNetwork implements GameController {
    private final int inputDim;
    private final int hiddenDim;
    private final int outputDim;
    private double[][] hiddenWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    protected double fitness;

    public FeedforwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim) {
        this.inputDim=inputDim;
        this.hiddenDim=hiddenDim;
        this.outputDim=outputDim;
        initializeParameters();
    }

    public FeedforwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim, double[] values) {
        this.inputDim=inputDim;
        this.hiddenDim=hiddenDim;
        this.outputDim=outputDim;
        loadParameters(values);
    }

    protected void initializeParameters(){
        hiddenWeights = new double[inputDim][hiddenDim];
        for(int i = 0; i<hiddenWeights.length; i++){
            for(int j = 0; j<hiddenWeights[i].length; j++){
                hiddenWeights[i][j]= new Random().nextDouble(-1,1);
            }
        }

        outputWeights = new double[hiddenDim][outputDim];
        for(int i = 0; i<outputWeights.length; i++){
            for(int j = 0; j<outputWeights[i].length; j++){
                outputWeights[i][j]= new Random().nextDouble(-1,1);
            }
        }

        hiddenBiases = new double[hiddenDim];
        for(int i = 0; i<hiddenBiases.length; i++){
            hiddenBiases[i] = new Random().nextDouble(-1,1);
        }

        outputBiases = new double[outputDim];
        for(int i = 0; i<outputBiases.length; i++){
            outputBiases[i] = new Random().nextDouble(-1,1);
        }
    }


    public int nextMove(int[] currentState){
        double[] out = forward(currentState);
        double value = out[0];
        int move = 0;
        for(int i =1; i<out.length; i++){
            if(out[i]>value){
                value = out[i];
                move = i;
            }
        }
        return move;
    }

    protected abstract void calculateFitness(int seed);

    protected double getFitness(){ return fitness; }

    public double[] forward(int[] inputValues) {
        if(inputValues.length!=inputDim) throw new IllegalArgumentException("Incorrect number of inputs");
        double[] hiddenLayer = new double[hiddenDim];
        for(int i = 0; i<hiddenDim; i++){
            for(int j = 0; j<inputDim; j++) {
                hiddenLayer[i] += (inputValues[j] * hiddenWeights[j][i]);
            }
            hiddenLayer[i] += hiddenBiases[i];
            hiddenLayer[i] = sigmoid(hiddenLayer[i]);
        }
        double[] output = new double[outputDim];
        for(int i = 0; i<outputDim; i++){
            for(int j = 0; j<hiddenLayer.length; j++) {
                output[i] += (hiddenLayer[j] * outputWeights[j][i]);
            }
            output[i] += outputBiases[i];
            output[i] = sigmoid(output[i]);
        }
        return output;
    }

    private static double sigmoid(double x){
        return 1/(1+Math.exp(-x));
    }
    public double leakyRelu(double x) {
        return Math.max(0.01 * x, x);
    }
    private static double tanh(double x){
        return Math.tanh(x);
    }
    private static double swish(double x) {
        return ((1 + Math.exp(-x)) + x * Math.exp(-x)) / Math.pow(1 + Math.exp(-x), 2);
    }

    protected void loadParameters(double[] values){
        if(values.length!=(inputDim*hiddenDim+hiddenDim+hiddenDim*outputDim+outputDim)){
            throw new IllegalArgumentException("Values length incorrect!");
        }
        int iter= 0;
        hiddenWeights = new double[inputDim][hiddenDim];
        for(int i = 0; i<hiddenWeights.length; i++){
            for(int j = 0; j<hiddenWeights[i].length; j++){
                hiddenWeights[i][j]=values[iter++];
            }
        }

        hiddenBiases = new double[hiddenDim];
        for(int i = 0; i<hiddenBiases.length; i++){
            hiddenBiases[i] = values[iter++];
        }

        outputWeights = new double[hiddenDim][outputDim];
        for(int i = 0; i<outputWeights.length; i++){
            for(int j = 0; j<outputWeights[i].length; j++){
                outputWeights[i][j]=values[iter++];
            }
        }

        outputBiases = new double[outputDim];
        for(int i = 0; i<outputBiases.length; i++){
            outputBiases[i] = values[iter++];
        }
    }

    public double[] getNeuralNetwork() {
        double[] neuralNetwork = new double[inputDim*hiddenDim+hiddenDim+hiddenDim*outputDim+outputDim];
        int iter = 0;
        for (double[] hiddenWeight : hiddenWeights) {
            for (double v : hiddenWeight) {
                neuralNetwork[iter++] = v;
            }
        }
        for (double hiddenBiase : hiddenBiases) {
            neuralNetwork[iter++] = hiddenBiase;
        }
        for (double[] outputWeight : outputWeights) {
            for (double v : outputWeight) {
                neuralNetwork[iter++] = v;
            }
        }
        for (double outputBiase : outputBiases) {
            neuralNetwork[iter++] = outputBiase;
        }
        return neuralNetwork;
    }

    @Override
    public String toString() {
        String result = "Neural Network: \nNumber of inputs: "
                + inputDim + "\n"
                + "Weights between input and hidden layer with " + hiddenDim + " neurons: \n";
        String hidden =
                "";
        for (int input = 0; input < inputDim; input++) {
            for (int i = 0; i < hiddenDim; i++) {
                hidden+= " w"+(input+1) + (i+1) +": "
                        + hiddenWeights[input][i] + "\n";
            }
        }
        result += hidden;
        String biasHidden = "Hidden biases: \n";
        for (int i = 0; i < hiddenDim; i++) {
            biasHidden += " b "+(i+1)+": " + hiddenBiases[i] +"\n";
        }
        result+= biasHidden;
        String output = "Weights between hidden and output layer with "
                + outputDim +" neurons: \n";
        for (int hiddenw = 0; hiddenw < hiddenDim; hiddenw++) {
            for (int i = 0; i < outputDim; i++) {
                output+= " w"+(hiddenw+1) +"o"+(i+1)+": "
                        + outputWeights[hiddenw][i] + "\n";
            }
        }
        result += output;
        String biasOutput = "Output biases: \n";
        for (int i = 0; i < outputDim; i++) {
            biasOutput += " bo"+(i+1)+": " + outputBiases[i] + "\n";
        }
        result+= biasOutput;
        return result;
    }

    protected void startThreads(Thread t1, Thread t2, Thread t3){
        t1.start();
        t2.start();
        t3.start();

        try {
            t1.join();
            t2.join();
            t3.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void resetFitness(){
        fitness=0;
    }

}
