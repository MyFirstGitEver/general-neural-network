package org.example.chooser;

import org.example.Vector;

public class Concater {
    private Vector[] inputs;

    public void setInputs(Vector[] inputs) throws Exception {
        if(inputs == null || inputs.length < 1) {
            throw new Exception("Concater requires inputs to be real or has length > 1!");
        }

        this.inputs = inputs;
    }

    public Vector of(int index) {
        if(index == 0) {
            return inputs[index];
        }

        return null; // by default
    }
}