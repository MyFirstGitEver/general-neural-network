package org.example.activators;

import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.networks.Neuron;

public class FeedForwardSoftmaxActivator implements Activator {
    @Override
    public double g(Neuron[] neuronSet, NumberGenerator backwardIds, int destination) {
        double total = 0;

        Integer source;

        double maxExponent = -Double.MAX_VALUE;
        while((source = backwardIds.next()) != null) {
            if(maxExponent < neuronSet[source].getValue()) {
                maxExponent = neuronSet[source].getValue();
            }
        }

        while ((source = backwardIds.next()) != null) {
            total += Math.exp(neuronSet[source].getValue() - maxExponent);
        }

        return Math.exp(neuronSet[destination].getValue() - maxExponent) / total;
    }

    @Override
    public double dg(Neuron[] neuronSet, NumberGenerator backwardIds, int source, int destination) {
        Integer sourceId;

        double maxExponent = -Double.MAX_VALUE;
        while((sourceId = backwardIds.next()) != null) {
            if(maxExponent < neuronSet[sourceId].getValue()) {
                maxExponent = neuronSet[sourceId].getValue();
            }
        }

        if(source == destination) {
            double term = g(neuronSet, backwardIds, source);
            return term * (1 - term);
        }

        double term1 = g(neuronSet, backwardIds, source);
        double term2 = g(neuronSet, backwardIds, destination);

        return - term1 * term2;
    }
}