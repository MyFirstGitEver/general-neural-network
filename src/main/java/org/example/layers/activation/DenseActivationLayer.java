package org.example.layers.activation;

import org.example.layers.Layer;
import org.example.layers.activation.ActivationLayer;
import org.example.networks.Neuron;
import org.example.activators.Activator;

public class DenseActivationLayer extends ActivationLayer {
    public DenseActivationLayer(int inputSize, int outputSize, Layer previousLayer, Activator activator) {
        super(inputSize, outputSize, previousLayer, activator);
    }

    @Override
    public ActivationLayer copy(Layer lastLayer) {
        return new DenseActivationLayer(X.length, Y.length, lastLayer, activator);
    }

    @Override
    public void buildXYRelations(Neuron[] X, Neuron[] Y, int... args) {
        Layer.EdgeBuilder.buildDense(Y, X);
    }
}