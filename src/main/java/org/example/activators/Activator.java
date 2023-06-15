package org.example.activators;

import org.example.chooser.NumberGenerator;
import org.example.networks.Neuron;

public interface Activator {
    double g(Neuron[] neuronSet, NumberGenerator backwardIds, int destination);
    double dg(Neuron[] neuronSet, NumberGenerator backwardIds, int source, int destination);
}