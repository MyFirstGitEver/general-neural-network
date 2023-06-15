package org.example.layers;

import org.example.Vector;
import org.example.layers.denselayer.activation.ActivationLayer;
import org.example.layers.denselayer.activation.DenseActivationLayer;
import org.example.layers.denselayer.weighted.DenseWeightedLayer;
import org.example.layers.denselayer.weighted.WeightedLayer;
import org.example.activators.Activator;
import org.example.neural.Matrix;

public class CompleteEdgeLayer {
    private final WeightedLayer weightedLayer;
    private final ActivationLayer activationLayer;

    public static final class Builder {
        public static CompleteEdgeLayer buildDenseDense(
                int inputSize,
                int outputSize,
                Activator activator,
                Layer lastLayer)
                throws Exception {

            DenseWeightedLayer weightedLayer = new DenseWeightedLayer(inputSize, outputSize, lastLayer);
            DenseActivationLayer activationLayer = new DenseActivationLayer(outputSize, outputSize, weightedLayer, activator);

            return new CompleteEdgeLayer(weightedLayer, activationLayer);
        }

        public static CompleteEdgeLayer buildDenseOne(
                int inputSize,
                int outputSize,
                Activator activator,
                Layer lastLayer)
                throws Exception {

            DenseWeightedLayer weightedLayer = new DenseWeightedLayer(inputSize, outputSize, lastLayer);
            OneToOneActivationLayer activationLayer =
                    new OneToOneActivationLayer(outputSize, outputSize, weightedLayer, activator);

            return new CompleteEdgeLayer(weightedLayer, activationLayer);
        }
    }

    public CompleteEdgeLayer(WeightedLayer weightedLayer, ActivationLayer activationLayer) throws Exception {
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

    public void setEigenDeltaForLast(Vector eigenDeltaForLast) throws Exception {
        if(eigenDeltaForLast.size() != activationLayer.eigenDelta.length) {
            throw new Exception("Eigen delta vector does fit with the edge layer's activation layer");
        }

        for(int i = 0; i< activationLayer.eigenDelta.length; i++) {
            activationLayer.eigenDelta[i] = eigenDeltaForLast.x(i);
        }
    }

    public void clip() {
        weightedLayer.clip();
    }

    public void load(Vector[] w, Vector b) {
        weightedLayer.load(w, b);
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

    public boolean valid(Matrix dW, Vector dB) {
        return weightedLayer.valid(dW, dB);
    }
}