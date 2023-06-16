package org.example.networks;

import org.example.Vector;
import org.example.activators.FeedForwardSoftmaxActivator;
import org.example.chooser.concaters.Concater;
import org.example.chooser.outputchoosers.OutputChooser;
import org.example.layers.CompleteForwarder;
import org.example.layers.Layer;
import org.example.loss.Loss;
import org.example.activators.Activator;
import org.example.others.DataGetter;

import java.io.BufferedReader;

public class FeedforwardNetwork {
    private final NeuronNetwork model;

    public FeedforwardNetwork(int[] sizes,
                                   Activator[] activators,
                                   Loss loss,
                                   DataGetter<Vector> xGetter,
                                   DataGetter<Vector> yGetter) throws Exception {

        CompleteForwarder[] layers = new CompleteForwarder[activators.length];
        layers[0] = inferFromActivator(sizes[0], sizes[1], activators[0], null);

        for(int i = 1;i < layers.length; i++) {
            layers[i] = inferFromActivator(
                    sizes[i],
                    sizes[i + 1],
                    activators[i],
                    layers[i - 1].getActivationLayer());
        }

        DataGetter<Vector[]> convertedXGetter = new DataGetter<>() {
            @Override
            public Vector[] at(int i) {
                return new Vector[] { xGetter.at(i) };
            }

            @Override
            public int size() {
                return xGetter.size();
            }
        };

        model = new NeuronNetwork(layers, loss, new Concater(), new OutputChooser() {
            @Override
            public boolean choose(int i) {
                return (i == activators.length - 1);
            }

            @Override
            public int count() {
                return 1;
            }
        }, convertedXGetter, yGetter);
    }

    public Vector output(Vector input) throws Exception {
        model.forward(new Vector[] { input });

        return model.getOutputs(1)[0];
    }

    public boolean train(double learningRate, int iteration, int batchSize, BufferedReader reader) throws Exception {
        return model.train(learningRate, iteration, batchSize, reader);
    }

    public void loadTestingToLayer(Vector[] W, Vector B, int layerId) {
        model.loadTestingToLayer(W, B, layerId, 0);
    }

    public CompleteForwarder inferFromActivator(
            int inputSize, int outputSize, Activator activator, Layer lastLayer) throws Exception {
        if(activator instanceof FeedForwardSoftmaxActivator) {
            return CompleteForwarder.Builder.buildDenseDense(inputSize, outputSize, activator, lastLayer);
        }

        return CompleteForwarder.Builder.buildDenseOne(inputSize, outputSize, activator, lastLayer);
    }
}