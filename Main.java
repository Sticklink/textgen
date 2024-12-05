import java.io.File;
import java.io.IOException;
import java.util.Scanner;

public class Main {
    public static final int NUM_INPUT_CHARS = 32;
    public static final int VECTOR_SIZE = 52;

    public static void main(String[] args) throws IOException {
        File dataFile = new File(System.getProperty("user.dir")+"/src/train.txt");

        Scanner fileReader = new Scanner(dataFile);

        System.out.println("Reading data...");

        int len=0;
        StringBuilder data = new StringBuilder();
        while (fileReader.hasNext()) {
            len++;
            data.append(fileReader.next().toLowerCase()).append(" ");
        }
        len = 100000;

        fileReader.close();
        System.out.println("Data read.");

        System.out.println("Parsing data...");

        double[][] inputs = new double[len-NUM_INPUT_CHARS-1][];
        double[][] solutions = new double[len-NUM_INPUT_CHARS-1][];

        for (int i=0;i<len-NUM_INPUT_CHARS-1;i++) {
            inputs[i] = new double[VECTOR_SIZE*NUM_INPUT_CHARS];
            solutions[i] = new double[VECTOR_SIZE];

            for (int j=0;j<NUM_INPUT_CHARS;j++) {
                char c = data.charAt(i+j);
                if (c=='\n') c=' ';
                int intVal = charToInt(c);

                inputs[i][j*VECTOR_SIZE+intVal] = 1.0;
            }

            int intVal = charToInt(data.charAt(i+NUM_INPUT_CHARS));
            solutions[i][intVal] = 1.0;
        }
        System.out.println("Data parsed.");

        System.out.println("Loading net...");
        NeuralNet net = new NeuralNet(new int[]{NUM_INPUT_CHARS*VECTOR_SIZE,128,128,VECTOR_SIZE}, inputs,solutions);
        net.load("net");
        System.out.println("Net loaded.");

        double avgCost;

        for (int i=0;;i++) {
            System.out.println(i*100000+" iters");
            avgCost = 0.0;
            for (int j=1000;j<1500;j++) {
                avgCost += net.getCost(inputs[j], solutions[j]);
            }
            avgCost*=0.002;

            System.out.println("Average cost: "+avgCost);

            String test = "Senator Jeff Bingaman of New Mexico brought this up to our " +
                    "attention, about the need to make sure there is a transition period " +
                    "between the ";

            for (int j=0;j<500;j++) {
                test += generate(net, test);
            }

            System.out.println("Output: "+test.substring(139));

            net.train(100000, 0.05);
            net.save("net");
        }
    }

    public static int charToInt(char c) {
        int index =  "abcdefghijklmnopqrstuvwxyz0123456789.,;!? ()'\"“”-’–".indexOf(c);
        if (index<0) return VECTOR_SIZE-1;
        return index;
    }

    public static char generate(NeuralNet net, String s) {
        double[] input = new double[NUM_INPUT_CHARS*VECTOR_SIZE];
        int len = s.length();
        for (int i=0;i<NUM_INPUT_CHARS;i++) {
            input[(NUM_INPUT_CHARS-i-1)*VECTOR_SIZE+charToInt(s.charAt(len-i-1))] = 1.0;
        }

        return net.getCharOutput(input);
    }
}
