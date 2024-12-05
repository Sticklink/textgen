import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class NeuralNet {
    private final int numLayers;
    private final Neuron[][] neurons;
    private final double[][] trainInputs;
    private final double[][] trainOutputs;
    private final int numTrainingExamples;

    public NeuralNet(int[] layerSizes, double[][] trainInputs, double[][] trainOutputs) {
        this.numLayers = layerSizes.length;
        this.neurons = new Neuron[this.numLayers][];
        for (int i=0;i<this.numLayers;i++) {
            int layerSize = layerSizes[i];
            neurons[i] = new Neuron[layerSize];
            for (int j=0;j<layerSize;j++) {
                neurons[i][j] = new Neuron(i==0?0:layerSizes[i-1]);
            }
        }
        this.trainInputs = trainInputs;
        this.trainOutputs = trainOutputs;
        this.numTrainingExamples = trainInputs.length;
    }
    public double[] getRawOutput(double[] input) {
        for (int i=0;i<input.length;i++) {
            neurons[0][i].setValue(input[i]);
        }
        for (int i=1;i<numLayers;i++) {
            for (Neuron n:neurons[i]) {
                n.calcValue(neurons[i-1]);
            }
        }

        int outputLength = neurons[numLayers-1].length;
        double[] outputVals = new double[outputLength];
        for (int i=0;i<outputLength;i++) {
            outputVals[i] = neurons[numLayers-1][i].getValue();
        }
        return outputVals;
    }

    public void train(int iters, double changeAmt) {
        Random r = new Random();
        double[][] batchNeuronChanges = new double[numLayers][];

        // init batchNeuronChanges layer sizes
        for (int layer=0;layer<numLayers;layer++){
            int len = neurons[layer].length;
            batchNeuronChanges[layer] = new double[len];
        }

        for (int i = 0; i < iters; i++) {

            int rand = r.nextInt(numTrainingExamples);

            double[] trainingExample = trainInputs[rand];
            double[] solution = trainOutputs[rand];

            getRawOutput(trainingExample);

            // init desiredNeuronChanges layer sizes

            for (int layer = 0; layer < numLayers; layer++) {
                int len = neurons[layer].length;
                batchNeuronChanges[layer] = new double[len];
            }

            // init desired changes
            for (int layer=numLayers-1;layer>1;layer--){
                int len = neurons[layer].length;
                for (int j=0;j<len;j++) {
                    Neuron n = neurons[layer][j];
                    if (layer == numLayers-1) {
                        batchNeuronChanges[layer][j] += (solution[j] - n.getValue());
                    }
                    for (int k=0;k<n.numWeights;k++) {
                        batchNeuronChanges[layer-1][k] += n.getWeight(k)*batchNeuronChanges[layer][j]/(neurons[layer].length);
                    }
                }
            }

            // do gradient descent
            for (int layer = 1; layer < numLayers; layer++) {
                for (int nInd = 0; nInd < neurons[layer].length; nInd++) {
                    Neuron n = neurons[layer][nInd];

                    double descentVal = Neuron.derSigmoid(n.getNonSigValue());
                    descentVal *= 2 * batchNeuronChanges[layer][nInd];
                    descentVal *= changeAmt;

                    n.changeBias(descentVal);

                    for (int j = 0; j < n.numWeights; j++) {
                        Neuron prevNeuron = neurons[layer - 1][j];
                        n.changeWeight(j, descentVal * prevNeuron.getValue());
                    }
                }
            }
        }
    }

    public void train(int iters) {
        train(iters, 0.01);
    }

    public double getCost(double[] input, double[] expected) {
        double[] actual = getRawOutput(input);
        int len = expected.length;
        double cost = 0;
        for (int i=0;i<len;i++) {
            double diff = (expected[i] - actual[i]);
            cost += diff*diff;
        }

        return cost;
    }

    public char getCharOutput(double[] input) {
        double[] rawOutput = getRawOutput(input);
        int maxInd = 0;
        for (int i=0;i<rawOutput.length;i++) {
            if (rawOutput[i]>rawOutput[maxInd]) maxInd = i;
        }
        return intToChar(maxInd);
    }

    public static char intToChar(int n) {
        return "abcdefghijklmnopqrstuvwxyz0123456789.,;!? ()'\"“”-’–$".charAt(n);
    }

    public void save(String fileName) throws IOException {
        String dir = System.getProperty("user.dir")+"/src/"+fileName+".txt";

        File saveFile = new File(dir);
        saveFile.createNewFile();

        FileWriter fw = new FileWriter(dir);
        for (Neuron[] nArr : neurons) {
            for (Neuron n : nArr) {
                for (int i=0;i<n.numWeights;i++) {
                    fw.write(n.getWeight(i) +"\n");
                }

                fw.write(n.getBias() +"\n");
            }
        }

        fw.close();
    }

    public void load(String fileName) throws IOException {
        String dir = System.getProperty("user.dir")+"/src/"+fileName+".txt";

        File saveFile = new File(dir);

        Scanner fr = null;
        try {
            fr = new Scanner(saveFile);
        } catch (FileNotFoundException e) {
            saveFile.createNewFile();
            fr = new Scanner(saveFile);
        }

        for (Neuron[] nArr : neurons) {
            for (Neuron n : nArr) {
                for (int i=0;i<n.numWeights;i++) {
                    n.setWeight(i,fr.nextDouble());
                }

                n.setBias(fr.nextDouble());
            }
        }
    }
}

