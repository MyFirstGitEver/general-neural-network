package org.example.layers.activation.rnn;

import org.example.activators.Activator;

import org.example.chooser.numbergenerators.SequentialGenerator;
import org.example.layers.Layer;
import org.example.layers.activation.ActivationLayer;
import org.example.networks.Neuron;

public class RNNActivationYLayer extends ActivationLayer {
    private final int ySize;

    private final int hSize;

    public RNNActivationYLayer(int inputSize, int outputSize, int ySize, int hSize,
                               Layer previousLayer, Activator activator) {
        super(inputSize, outputSize, previousLayer, activator);

        this.ySize = ySize;
        this.hSize = hSize;
    }

    @Override
    public void buildXYRelations(Neuron[] X, Neuron[] Y) {
        for(int i=0;i<ySize;i++) {
            X[i].setForwardNeurons(new SequentialGenerator(0, ySize));
            Y[i].setBackwardNeurons(new SequentialGenerator(0, ySize));
        }

        for(int i=ySize;i<ySize + hSize;i++) {
            X[i].setForwardNeurons(new SequentialGenerator(i, i + 1));
        }
    }

    @Override
    public ActivationLayer copy(Layer lastLayer) {
        return new RNNActivationYLayer(X.length, Y.length, ySize, hSize, lastLayer, activator);
    }
}