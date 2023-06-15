package org.example.layers.denselayer.activation;

import org.example.layers.Layer;
import org.example.networks.Neuron;
import org.example.activators.Activator;

public class DenseActivationLayer extends ActivationLayer {
    public DenseActivationLayer(int inputSize, int outputSize, Layer previousLayer, Activator activator) {
        super(inputSize, outputSize, previousLayer, activator);
    }

    @Override
    public void buildXYRelations(Neuron[] X, Neuron[] Y) {
        Layer.EdgeBuilder.buildDense(Y, X);
    }
}