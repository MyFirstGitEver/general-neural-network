package org.example.networks;

import org.example.TestingObject;
import org.example.Vector;
import org.example.chooser.concaters.RNNConcater;
import org.example.chooser.numbergenerators.SequentialGenerator;
import org.example.chooser.outputchoosers.RNNOutputProcessor;
import org.example.layers.CompleteForwarder;

import org.example.loss.CrossEntropy;
import org.example.neural.Matrix;
import org.example.others.DataGetter;

import java.util.List;

public class RNN {
    private final NeuronNetwork model;

    public RNN(
            int inputSize,
            int priorKnowledgeSize,
            int outputSize,
            int timeStep,
            DataGetter<Vector[]> xGetter,
            DataGetter<Vector> yGetter) throws Exception {
        CompleteForwarder[] edgeLayers = new CompleteForwarder[2 * timeStep];

        edgeLayers[0] = CompleteForwarder.Builder.buildFirstRNNLayer(inputSize, priorKnowledgeSize, outputSize);
        edgeLayers[1] = CompleteForwarder.Builder.buildSecondRNNLayer(inputSize, priorKnowledgeSize, outputSize,
                edgeLayers[0].getActivationLayer());

        for(int i=1;i<timeStep;i++) {
            edgeLayers[2 * i] = edgeLayers[0].sharedLayersForwarder(edgeLayers[2 * i - 1].getActivationLayer());
            edgeLayers[2 * i + 1] = edgeLayers[1].sharedLayersForwarder(edgeLayers[2 * i].getActivationLayer());
        }

        model = new NeuronNetwork(
                edgeLayers, new CrossEntropy(), new RNNConcater(),
                new RNNOutputProcessor(timeStep, outputSize), xGetter, yGetter, new SequentialGenerator(0, 2));
    }

    public boolean train(double learningRate, int iteration, int batchSize, boolean printCost,
                         List<TestingObject> tests, List<TestingObject> params) throws Exception {
        return model.train(learningRate, iteration, batchSize, 10, tests, params, printCost);
    }

    public Vector output(Vector[] inputs) throws Exception {
        model.forward(inputs);

        return model.getOutputs(1)[0];
    }

    public Vector[] outputs(Vector[] inputs) throws Exception {
        model.forward(inputs);

        return model.getOutputs(model.outputCount());
    }

    public void loadParameters(Matrix Whh, Matrix Whx, Matrix Wyh, Vector bh, Vector by) {
        // input index offset by ySize
        model.loadTestingToLayer(Whh.vectorize(true), bh, 0, by.size());

        // input index offset by ySize + hSize
        model.loadTestingToLayer(Whx.vectorize(true), null, 0, by.size() + bh.size());

        model.loadTestingToLayer(Wyh.vectorize(true), by, 1, 0);
    }
}