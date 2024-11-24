package utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

public class IO_Utils {
    public static void writeToFile(String fileName, String content){
        PrintWriter writer = null;
        try {
            writer = new PrintWriter(new File(fileName));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        writer.print(content);
        writer.close();
    }

    public static String readFromFile(String inputFileName){
        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(inputFileName));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        return scanner.nextLine();
    }

    public static double[] fromString(String string) {
        String[] strings = string.replace("[", "").replace("]", "").split(", ");
        double result[] = new double[strings.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Double.parseDouble(strings[i]);
        }
        return result;
    }
}
