package org.example.neural;

import org.example.Vector;

public class LinearActivation extends ActivationFunction {
    public LinearActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    public Vector out(Vector z) {
        return z;
    }

    @Override
    public Vector derivativeByZ(Vector z, Vector y) {
        return new Vector(z.size(), 1.0f);
    }

}