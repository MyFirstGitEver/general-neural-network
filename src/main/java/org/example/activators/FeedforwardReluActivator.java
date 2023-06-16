package org.example.activators;

import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.networks.Neuron;

public class FeedforwardReluActivator implements Activator {
    @Override
    public double g(Neuron[] neuronSet, NumberGenerator backwardIds, int destination) {
        return Math.max(neuronSet[destination].getValue(), 0.0);
    }

    @Override
    public double dg(Neuron[] neuronSet, NumberGenerator backwardIds, int source, int destination) {
        return neuronSet[destination].getValue() <= 0.0 ? 0.0 : 1.0;
    }
}