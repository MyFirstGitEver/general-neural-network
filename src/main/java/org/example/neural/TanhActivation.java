package org.example.neural;

import org.example.Vector;

public class TanhActivation extends ActivationFunction {
    public TanhActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    public Vector out(Vector z) {
        Vector a = new Vector(z.size());

        for(int i=0;i<a.size();i++) {
            a.setX(i, 2 * sigmoid(z.x(i) * 2) - 1);
        }

        return a;
    }

    @Override
    public Vector derivativeByZ(Vector z, int aPos) {
        Vector a = new Vector(z.size());

        double sig = sigmoid(2 * z.x(aPos));
        a.setX(aPos, 4 * sig * (1 - sig));

        return a;
    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
}