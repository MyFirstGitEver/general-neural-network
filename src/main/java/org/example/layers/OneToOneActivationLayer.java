package org.example.layers;

import org.example.activators.Activator;
import org.example.chooser.numbergenerators.SequentialGenerator;
import org.example.layers.activation.ActivationLayer;
import org.example.networks.Neuron;

public class OneToOneActivationLayer extends ActivationLayer {
    public OneToOneActivationLayer(int inputSize, int outputSize, Layer previousLayer, Activator activator) {
        super(inputSize, outputSize, previousLayer, activator);
    }

    @Override
    public ActivationLayer copy(Layer lastLayer) {
        return new OneToOneActivationLayer(X.length, Y.length, lastLayer, activator);
    }

    @Override
    public void buildXYRelations(Neuron[] X, Neuron[] Y) {
        for(int i=0;i<X.length;i++) {
            X[i].setForwardNeurons(new SequentialGenerator(i, i + 1));
        }

        for(int i=0;i<X.length;i++) {
            Y[i].setBackwardNeurons(new SequentialGenerator(i, i + 1));
        }
    }
}