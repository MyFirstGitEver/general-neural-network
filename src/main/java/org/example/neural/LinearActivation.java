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
    public Vector derivativeByZ(Vector z, int aPos) {
        Vector dz = new Vector(z.size());
        dz.setX(aPos, 1.0);

        return dz;
    }
}