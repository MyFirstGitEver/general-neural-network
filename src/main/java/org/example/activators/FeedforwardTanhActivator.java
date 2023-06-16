package org.example.activators;

import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.networks.Neuron;

public class FeedforwardTanhActivator implements Activator {
    @Override
    public double g(Neuron[] neuronSet, NumberGenerator backwardIds, int destination) {
        double z = neuronSet[destination].getValue();

        return 2 * sigmoid(2 * z) - 1;
    }

    @Override
    public double dg(Neuron[] neuronSet, NumberGenerator backwardIds, int source, int destination) {
        double z = neuronSet[destination].getValue();
        double sig = sigmoid(2 * z);

        return 4 * sig * (1 - sig);
    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
}