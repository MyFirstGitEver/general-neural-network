package org.example.layers.weighted.rnn;

import org.example.chooser.numbergenerators.NPlusKGenerator;
import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.chooser.numbergenerators.SequentialGenerator;
import org.example.layers.Layer;
import org.example.layers.weighted.WeightedLayer;
import org.example.networks.Neuron;
import org.example.others.Learnable;

public class RNNWeightedYLayer extends WeightedLayer {
    private final int ySize;

    public RNNWeightedYLayer(
            int inputSize,
            int outputSize,
            Layer previousLayer,
            int ySize) {
        super(inputSize, outputSize, previousLayer);

        this.ySize = ySize;
    }

    public RNNWeightedYLayer(WeightedLayer weightedLayer, Layer previousLayer, int ySize) {
        super(weightedLayer, previousLayer);

        this.ySize = ySize;
    }

    @Override
    public void buildXYRelations(Neuron[] X, Neuron[] Y) {
        for(int i=0;i<X.length;i++) {
            X[i].setForwardNeurons(new NPlusKGenerator(0, ySize, ySize + (i + 1)));
        }

        for(int i=0;i<ySize;i++) {
            Y[i].setBackwardNeurons(new SequentialGenerator(0, X.length));
        }
    }

    @Override
    public void buildWeights() {
        for(int destination=0;destination<Y.length;destination++) {
            NumberGenerator generator = Y[destination].getBackwardNeurons();

            if(destination >= ySize) {
                wParamsOf.put(hash(generator.next(), destination), new Learnable(1.0));
                continue;
            }

            Integer source;

            while((source = generator.next()) != null) {
                wParamsOf.put(hash(source, destination), new Learnable(Math.random()));
            }
        }
    }

    @Override
    public WeightedLayer copy(Layer lastLayer) {
        return new RNNWeightedYLayer(this, lastLayer, ySize);
    }

    @Override
    public void buildBiases() {
        for(int i = 0;i < bParams.length; i++) {
            if(i >= ySize) {
                bParams[i] = new Learnable(0.0);
            }
            else {
                bParams[i] = new Learnable(Math.random());
            }
        }
    }
}