import algorithms.*;
import breakout.Breakout;
import pacman.Pacman;
import utils.Commons;
import utils.IO_Utils;
import java.io.IOException;
import java.util.Arrays;


public class Main {

    public static void main(String[] args){
        breakoutSearch();
        //pacmanSearch();
    }

    public static void breakoutSearch(){
        int seed = 99;
        GeneticAlghorithm gn = new BreakOutGA(seed,20000);
        FeedforwardNeuralNetwork nn = gn.beginSearch();
        IO_Utils.writeToFile("melhorBreakout.txt", Arrays.toString(nn.getNeuralNetwork()));
        Breakout b = new Breakout(nn,99);
        System.out.println(Arrays.toString(nn.getNeuralNetwork()));
    }

    public static void pacmanSearch(){
        int seed = 99;
        GeneticAlghorithm gn = new PacManGA(seed,5000);
        FeedforwardNeuralNetwork nn = gn.beginSearch();
        IO_Utils.writeToFile("melhorPacman.txt", Arrays.toString(nn.getNeuralNetwork()));
        Pacman p = new Pacman(nn,true,99);
        System.out.println(Arrays.toString(nn.getNeuralNetwork()));
    }


}
