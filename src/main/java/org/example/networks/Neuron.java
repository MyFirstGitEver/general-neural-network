package org.example.networks;

import org.example.chooser.numbergenerators.NumberGenerator;

public class Neuron {
    private double value;
    private NumberGenerator forwardNeurons, backwardNeurons;

    public Neuron(double value) {
        this.value = value;
    }


    public double getValue() {
        return value;
    }

    public NumberGenerator getForwardNeurons() {
        return forwardNeurons;
    }

    public NumberGenerator getBackwardNeurons() {
        return backwardNeurons;
    }

    public void setForwardNeurons(NumberGenerator forwardNeurons) {
        this.forwardNeurons = forwardNeurons;
    }

    public void setBackwardNeurons(NumberGenerator backwardNeurons) {
        this.backwardNeurons = backwardNeurons;
    }

    public void setValue(double value) {
        this.value = value;
    }
}