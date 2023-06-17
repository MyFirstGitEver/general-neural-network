package org.example.layers.activation.rnn;

import org.example.activators.Activator;

import org.example.chooser.numbergenerators.SequentialGenerator;
import org.example.layers.Layer;
import org.example.layers.activation.ActivationLayer;
import org.example.networks.Neuron;

public class RNNActivationYLayer extends ActivationLayer {
    private final int ySize;
    private final int hSize;

    public RNNActivationYLayer(int inputSize, int outputSize, Layer previousLayer, Activator activator, int... args) {
        super(inputSize, outputSize, previousLayer, activator, args);

        this.ySize = args[0];
        this.hSize = args[1];
    }

    @Override
    protected void buildXYRelations(Neuron[] X, Neuron[] Y, int... args) {
        int ySize = args[0];
        int hSize = args[1];

        for(int i=0;i<ySize;i++) {
            X[i].setForwardNeurons(new SequentialGenerator(0, ySize));
            Y[i].setBackwardNeurons(new SequentialGenerator(0, ySize));
        }

        for(int i=ySize;i<ySize + hSize;i++) {
            Y[i].setBackwardNeurons(new SequentialGenerator(i, i + 1));
        }
    }

    @Override
    public ActivationLayer copy(Layer lastLayer) {
        return new RNNActivationYLayer(X.length, Y.length, lastLayer, activator, ySize, hSize);
    }
}