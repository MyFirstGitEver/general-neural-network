package org.example.activators.rnn;

import org.example.activators.Activator;
import org.example.activators.FeedForwardSoftmaxActivator;
import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.networks.Neuron;

public class RNNSoftmaxActivator extends FeedForwardSoftmaxActivator {
    @Override
    public double g(Neuron[] neuronSet, NumberGenerator backwardIds, int destination) {
        if(backwardIds.count() == 1) {
            int next = backwardIds.next();

            backwardIds.reset();
            return neuronSet[next].getValue();
        }

        return super.g(neuronSet, backwardIds, destination);
    }
}