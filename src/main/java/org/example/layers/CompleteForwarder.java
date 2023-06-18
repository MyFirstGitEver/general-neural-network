package org.example.layers;

import org.example.Vector;
import org.example.activators.FeedforwardTanhActivator;
import org.example.activators.rnn.RNNSoftmaxActivator;
import org.example.layers.activation.ActivationLayer;
import org.example.layers.activation.DenseActivationLayer;
import org.example.layers.activation.rnn.RNNActivationYLayer;
import org.example.layers.weighted.DenseWeightedLayer;
import org.example.layers.weighted.WeightedLayer;
import org.example.activators.Activator;
import org.example.layers.weighted.rnn.RNNWeightedHLayer;
import org.example.layers.weighted.rnn.RNNWeightedYLayer;
import org.example.neural.Matrix;

public class CompleteForwarder {
    private final WeightedLayer weightedLayer;
    private final ActivationLayer activationLayer;
    public static final class Builder {
        public static CompleteForwarder buildDenseDense(
                int inputSize,
                int outputSize,
                Activator activator,
                Layer lastLayer)
                throws Exception {

            DenseWeightedLayer weightedLayer = new DenseWeightedLayer(inputSize, outputSize, lastLayer);
            DenseActivationLayer activationLayer = new DenseActivationLayer(
                    outputSize,
                    outputSize,
                    weightedLayer ,activator);

            return new CompleteForwarder(weightedLayer, activationLayer);
        }

        public static CompleteForwarder buildDenseOne(
                int inputSize,
                int outputSize,
                Activator activator,
                Layer lastLayer)
                throws Exception {

            DenseWeightedLayer weightedLayer = new DenseWeightedLayer(inputSize, outputSize, lastLayer);
            OneToOneActivationLayer activationLayer =
                    new OneToOneActivationLayer(outputSize, outputSize, weightedLayer, activator);

            return new CompleteForwarder(weightedLayer, activationLayer);
        }

        public static CompleteForwarder buildFirstRNNLayer(
                int inputSize, int priorKnowledgeSize, int outputSize) throws Exception {

            WeightedLayer weightedLayer = new RNNWeightedHLayer(
                    inputSize + priorKnowledgeSize + outputSize,
                    priorKnowledgeSize,
                    null,
                    outputSize);

            ActivationLayer activationLayer =
                    new OneToOneActivationLayer(
                            priorKnowledgeSize,
                            priorKnowledgeSize,
                            weightedLayer,
                            new FeedforwardTanhActivator());

            return new CompleteForwarder(weightedLayer, activationLayer);
        }
        public static CompleteForwarder buildSecondRNNLayer(
                int inputSize,
                int priorKnowledgeSize,
                int outputSize,
                ActivationLayer hActivationLayer) throws Exception {

            WeightedLayer weightedLayer = new RNNWeightedYLayer(
                    priorKnowledgeSize, priorKnowledgeSize + outputSize, hActivationLayer);

            ActivationLayer activationLayer = new RNNActivationYLayer(
                    priorKnowledgeSize + outputSize,
                    priorKnowledgeSize + inputSize + outputSize,
                    weightedLayer,
                    new RNNSoftmaxActivator(), outputSize, priorKnowledgeSize);

            return new CompleteForwarder(weightedLayer, activationLayer);
        }
    }

    public CompleteForwarder(WeightedLayer weightedLayer, ActivationLayer activationLayer) throws Exception {
        this.weightedLayer = weightedLayer;
        this.activationLayer = activationLayer;

        if(weightedLayer.getY() != activationLayer.getX()) {
            throw new Exception("This layer is not complete!");
        }
    }

    public void forward() {
        weightedLayer.forward();
        activationLayer.forward();
    }

    public void concatInput(Vector input) throws Exception {
        weightedLayer.concat(input);
    }

    public void backward() {
        activationLayer.backward();
        weightedLayer.backward();
    }

    public void update(double learningRate, int outputCount, int datasetSize) {
        weightedLayer.update(learningRate, outputCount, datasetSize);
    }

    public void setEigenDeltas(Vector eigenDeltaForLast) {
        for(int i = 0; i< eigenDeltaForLast.size(); i++) {
            activationLayer.eigenDelta[i] = eigenDeltaForLast.x(i);
        }
    }

    public void clip() {
        weightedLayer.clip();
    }

    public void load(Vector[] w, Vector b, int offset) {
        weightedLayer.load(w, b, offset);
    }

    public int getASize() {
        return activationLayer.Y.length;
    }

    public int getXSize() {
        return weightedLayer.X.length;
    }

    public Vector getFinalOutput() {
        Vector output = new Vector(activationLayer.Y.length);

        for(int i=0;i<output.size();i++) {
            output.setX(i, activationLayer.Y[i].getValue());
        }

        return output;
    }

    public ActivationLayer getActivationLayer() {
        return activationLayer;
    }

    public boolean valid(Matrix dW, Vector dB, int offset) {
        return weightedLayer.valid(dW, dB, offset);
    }

    public boolean validParams(Matrix W, Vector B, int offset) {
        return weightedLayer.validParams(W, B, offset);
    }

    public CompleteForwarder sharedLayersForwarder(Layer lastLayer) throws Exception {
        WeightedLayer weightedLayer = this.weightedLayer.copy(lastLayer);

        return new CompleteForwarder(weightedLayer, activationLayer.copy(weightedLayer));
    }
}