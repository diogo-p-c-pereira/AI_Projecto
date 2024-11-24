package algorithms;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

public abstract class GeneticAlghorithm {
    protected static final int N_POPULATION = 100;
    private static final double P_MUTACAO = 0.80;
    private static final double SELECAO = 0.20;
    private final int MAX_GERACOES;
    public int SEED;

    protected FeedforwardNeuralNetwork[] population;
    public GeneticAlghorithm(int seed, int geracoes){
        this.SEED = seed;
        this.MAX_GERACOES = geracoes;
        createPopulation();
    }

    protected abstract void createPopulation();

    protected abstract void showGenResult(FeedforwardNeuralNetwork genBest, int i);

    protected abstract FeedforwardNeuralNetwork createNewNN(double[] nn);

    public FeedforwardNeuralNetwork beginSearch(){
        for(int i = 0; i<MAX_GERACOES; i++){
            if((i%200)==0){
                FeedforwardNeuralNetwork genBest = best();
                showGenResult(genBest,i);
            }
            if((i%400==0) && i!=0) {
                SEED = SEED << 1;
                resetFitnessValues();
            }
            FeedforwardNeuralNetwork[] nns = eliteSelection(population);
            nns = crossover(nns,(int)(N_POPULATION*SELECAO),(int)(N_POPULATION-(N_POPULATION*P_MUTACAO)));
            for(int k = (int)(N_POPULATION-(N_POPULATION*P_MUTACAO)); k<N_POPULATION; k++){
                nns[k]=mutation(nns[(int)(Math.random()*(N_POPULATION*SELECAO))]);
            }
            population=nns;
        }
        return best();
    }

    private FeedforwardNeuralNetwork best(){
        FeedforwardNeuralNetwork best = population[0];
        if(best.getFitness()==0){
            best.calculateFitness(SEED);
        }
        for(int i = 0; i<N_POPULATION; i++){
            FeedforwardNeuralNetwork nn = population[i];
            if(nn.getFitness()==0){
                nn.calculateFitness(SEED);
            }
            if(best.getFitness()<nn.getFitness()){
                best = nn;
            }
        }
        return best;
    }

    private FeedforwardNeuralNetwork[] eliteSelection(FeedforwardNeuralNetwork[] nn){
        Arrays.sort(nn, (n1, n2) -> {
            if(n1.getFitness()==0){
                n1.calculateFitness(SEED);
            }
            if(n2.getFitness()==0){
                n2.calculateFitness(SEED);
            }
            return Double.compare(n2.getFitness(), n1.getFitness());
        });
        FeedforwardNeuralNetwork[] selected = new FeedforwardNeuralNetwork[nn.length];
        for(int i = 0; i<(N_POPULATION*SELECAO); i++){
            selected[i] = nn[i];
        }
        return selected;
    }

    private FeedforwardNeuralNetwork[] selection(FeedforwardNeuralNetwork[] nn) {
        FeedforwardNeuralNetwork[] selected = new FeedforwardNeuralNetwork[N_POPULATION];
        for(int i = 0; i<N_POPULATION*SELECAO; i++){
            FeedforwardNeuralNetwork nn1 = nn[(int)(Math.random()*nn.length)];
            FeedforwardNeuralNetwork nn2= nn[(int)(Math.random()*nn.length)];
            if(nn1.getFitness()==0){
                nn1.calculateFitness(SEED);
            }
            if(nn2.getFitness()==0){
                nn2.calculateFitness(SEED);
            }
            if(nn1.getFitness() > nn2.getFitness()) {
                selected[i] = nn1;
            }else{
                selected[i] = nn2;
            }
        }
        return selected;
    }

    private FeedforwardNeuralNetwork[] crossover(FeedforwardNeuralNetwork[] nn, int start, int end) {
        for(int i = start; i<end; i=i+2){
            double[] a1 = nn[new Random().nextInt(start)].getNeuralNetwork();
            double[] a2 = nn[new Random().nextInt(start)].getNeuralNetwork();
            double[] n1 = new double[a1.length];
            double[] n2 = new double[a1.length];
            int crosspoint = new Random().nextInt(a1.length);
            for(int j = 0; j <a1.length; j++){
                if(j <crosspoint){
                    n1[j] = a1[j];
                    n2[j] = a2[j];
                }else{
                    n1[j] = a2[j];
                    n2[j] = a1[j];
                }
            }
            nn[i] = createNewNN(n1);
            nn[i+1] = createNewNN(n2);
        }
        return nn;
    }

    private FeedforwardNeuralNetwork mutation(FeedforwardNeuralNetwork nn){
        double[] nnC = nn.getNeuralNetwork();
        int r = new Random().nextInt(nnC.length);
        nnC[r] = nnC[r] + new Random().nextDouble(-1, 1);
        return createNewNN(nnC);
    }


    private FeedforwardNeuralNetwork bitSwapMutation(FeedforwardNeuralNetwork nn){
        double[] nnC = nn.getNeuralNetwork();
        Random random = new Random();
        for(int g=0; g<=5; g++) {
            int index1 = random.nextInt(nnC.length);
            int index2 = random.nextInt(nnC.length);

            double temp = nnC[index1];
            nnC[index1] = nnC[index2];
            nnC[index2] = temp;
        }
        return createNewNN(nnC);
    }

    private FeedforwardNeuralNetwork bitFlipMutation(FeedforwardNeuralNetwork nn){
        double[] nnC = nn.getNeuralNetwork();
        for(int g=0; g<=15;g++) {
            nnC[new Random().nextInt(nnC.length)] = new Random().nextDouble(-1, 1);
        }
        return createNewNN(nnC);
    }

    private void resetFitnessValues(){
        for(FeedforwardNeuralNetwork nn: population){
            nn.resetFitness();
        }
    }

}
