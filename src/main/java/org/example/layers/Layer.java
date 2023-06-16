package org.example.layers;

import org.example.chooser.numbergenerators.SequentialGenerator;
import org.example.layers.weighted.WeightedLayer;
import org.example.networks.Neuron;
import org.example.Vector;

public abstract class Layer {
    public final static class EdgeBuilder {
        public static void buildDense(Neuron[] Y, Neuron[] X) {
            for (Neuron neuron : Y) {
                neuron.setBackwardNeurons(new SequentialGenerator(0, X.length));
            }

            for (Neuron x : X) {
                x.setForwardNeurons(new SequentialGenerator(0, Y.length));
            }
        }
    }

    protected final double[] eigenDelta;
    protected final Neuron[] X;
    protected final Neuron[] Y;

    protected final Layer previousLayer;

    public Layer(int inputSize, int outputSize, Layer previousLayer) {
        if(previousLayer != null) {
            X = previousLayer.Y;
        }
        else {
            X = new Neuron[inputSize];

            for(int i=0;i<X.length;i++) {
                X[i] = new Neuron(0);
            }
        }

        Y = new Neuron[outputSize];
        for(int i=0;i<Y.length;i++) {
            Y[i] = new Neuron(0);
        }

        buildXYRelations(X, Y);

        this.previousLayer = previousLayer;
        this.eigenDelta = new double[this.Y.length];
    }

    abstract public void forward();
    abstract public void backward();
    abstract public void buildXYRelations(Neuron[] X, Neuron[] Y);

    public Neuron[] getX() {
        return X;
    }

    public Neuron[] getY() {
        return Y;
    }

    public void concat(Vector input) throws Exception {
        if(input.size() > X.length) {
            throw new Exception("Neuron set is too short to concat");
        }

        int index = input.size() - 1;
        for(int j = X.length-1;j>=X.length-input.size();j--) {
            X[j].setValue(input.x(index));

            index--;
        }
    }

    public double[] getEigenDelta() {
        return eigenDelta;
    }
}