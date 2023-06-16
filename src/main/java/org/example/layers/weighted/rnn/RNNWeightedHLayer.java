package org.example.layers.weighted.rnn;

import org.example.chooser.numbergenerators.SequentialGenerator;
import org.example.layers.Layer;
import org.example.layers.weighted.WeightedLayer;
import org.example.networks.Neuron;

public class RNNWeightedHLayer extends WeightedLayer {
    private final int ySize;

    public RNNWeightedHLayer(
            int inputSize,
            int outputSize,
            Layer previousLayer,
            int ySize) {
        super(inputSize, outputSize, previousLayer);

        this.ySize = ySize;
    }

    public RNNWeightedHLayer(WeightedLayer weightedLayer, Layer previousLayer, int ySize) {
        super(weightedLayer, previousLayer);

        this.ySize = ySize;
    }

    @Override
    public WeightedLayer copy(Layer lastLayer) {
        return new RNNWeightedHLayer(this, lastLayer, ySize);
    }

    @Override
    public void buildXYRelations(Neuron[] X, Neuron[] Y) {
        for(int i=ySize;i<X.length;i++) {
            X[i].setForwardNeurons(new SequentialGenerator(0, Y.length));
        }

        for (Neuron neuron : Y) {
            neuron.setBackwardNeurons(new SequentialGenerator(ySize, X.length));
        }
    }
}