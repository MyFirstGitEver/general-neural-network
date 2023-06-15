package org.example.layers.denselayer.weighted;

import org.example.layers.Layer;
import org.example.networks.Neuron;

public class DenseWeightedLayer extends WeightedLayer {
    public DenseWeightedLayer(int inputSize, int outputSize, Layer previousLayer) {
        super(inputSize, outputSize, previousLayer);
    }

    @Override
    public void buildXYRelations(Neuron[] X, Neuron[] Y) {
        Layer.EdgeBuilder.buildDense(Y, X);
    }
}