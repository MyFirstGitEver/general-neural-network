package org.example.activators;

import org.example.chooser.NumberGenerator;
import org.example.networks.Neuron;

public class FeedForwardLinearActivator implements Activator{
    @Override
    public double g(Neuron[] neuronSet, NumberGenerator backwardIds, int destination) {
        return neuronSet[destination].getValue();
    }

    @Override
    public double dg(Neuron[] neuronSet, NumberGenerator backwardIds, int source, int destination) {
        return 1;
    }
}
