package org.example.layers.weighted;

import org.example.layers.Layer;
import org.example.networks.Neuron;

public class DenseWeightedLayer extends WeightedLayer {
    public DenseWeightedLayer(int inputSize, int outputSize, Layer previousLayer) {
        super(inputSize, outputSize, previousLayer);
    }

    public DenseWeightedLayer(WeightedLayer weightedLayer, Layer previousLayer) {
        super(weightedLayer, previousLayer);
    }

    @Override
    protected void buildXYRelations(Neuron[] X, Neuron[] Y, int... args) {
        Layer.EdgeBuilder.buildDense(Y, X);
    }

    @Override
    public WeightedLayer copy(Layer lastLayer) {
        return new DenseWeightedLayer(this, lastLayer);
    }
}