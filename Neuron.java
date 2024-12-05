public class Neuron {
    private final double[] weights;
    public final int numWeights;
    private double bias;
    private double value;
    private double nonSigValue;

    public Neuron(double[] weights, int numWeights, double bias) {
        this.weights = weights;
        this.numWeights = numWeights;
        this.bias = bias;
    }

    public Neuron(int numWeights) {
        this.weights = new double[numWeights];
        for (int i=0;i<numWeights;i++) {
            this.weights[i] = 2.0*Math.random()-1.0;
        }
        this.numWeights = numWeights;
        this.bias = 0.0;
    }

    public static double sigmoid(double x) {
        return 1.0/(1.0+Math.exp(-x));
    }

    public static double derSigmoid(double x) {
        double sig = sigmoid(x);
        return sig*(1-sig);
    }

    public void calcValue(Neuron[] neurons) {
        double sum = 0;
        for (int i=0;i<numWeights;i++) {
            sum += weights[i]*neurons[i].getValue();
        }
        nonSigValue = sum+bias;
        value = sigmoid(nonSigValue);
    }

    public void setValue(double value) {
        this.value=value;
    }

    public double getValue() {
        return value;
    }

    public Neuron copy() {
        return new Neuron(this.weights.clone(), this.numWeights, this.bias);
    }

    public double getWeight(int index) {
        return this.weights[index];
    }
    public void setWeight(int index, double val) {
        this.weights[index] = val;
    }

    public double getBias() {
        return bias;
    }
    public void setBias(double val) {
        bias = val;
    }

    public double getNonSigValue() {
        return nonSigValue;
    }

    public void changeWeight(int index, double amt) {
        this.weights[index] += amt;
    }

    public void changeBias(double amt) {
        this.bias += amt;
    }
}
