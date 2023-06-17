package org.example.layers.weighted.rnn;

import org.example.chooser.numbergenerators.NPlusKGenerator;
import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.chooser.numbergenerators.SequentialGenerator;
import org.example.layers.Layer;
import org.example.layers.weighted.WeightedLayer;
import org.example.networks.Neuron;
import org.example.others.Learnable;

public class RNNWeightedYLayer extends WeightedLayer {
    public RNNWeightedYLayer(int inputSize, int outputSize, Layer previousLayer) {
        super(inputSize, outputSize, previousLayer);
    }

    public RNNWeightedYLayer(WeightedLayer weightedLayer, Layer previousLayer) {
        super(weightedLayer, previousLayer);
    }

    @Override
    protected void buildXYRelations(Neuron[] X, Neuron[] Y, int... args) {
        int ySize = Y.length - X.length;

        for (Neuron x : X) {
            x.setForwardNeurons(new SequentialGenerator(0, ySize));
        }

        for(int i=0;i<ySize;i++) {
            Y[i].setBackwardNeurons(new SequentialGenerator(0, X.length));
        }

        for(int i=ySize;i<Y.length;i++) {
            Y[i].setBackwardNeurons(new SequentialGenerator(i - ySize, i - ySize + 1));
        }
    }

    @Override
    protected void buildWeights() {
        int ySize = Y.length - X.length;

        for(int destination=0;destination<ySize;destination++) {
            NumberGenerator generator = Y[destination].getBackwardNeurons();
            Integer source;

            while((source = generator.next()) != null) {
                wParamsOf.put(hash(source, destination), new Learnable(Math.random()));
            }
        }

        for(int destination=ySize;destination< Y.length;destination++) {
            wParamsOf.put(hash(destination - ySize, destination), new Learnable(1.0));
        }
    }

    @Override
    protected void buildBiases() {
        int ySize = Y.length - X.length;

        for(int i = 0;i < ySize; i++) {
            bParams[i] = new Learnable(Math.random());
        }

        for(int i=ySize;i<bParams.length;i++) {
            bParams[i] = new Learnable(0.0);
        }
    }

    @Override
    public WeightedLayer copy(Layer lastLayer) {
        return new RNNWeightedYLayer(this, lastLayer);
    }
}