package org.example.neural;

import org.example.Vector;

public abstract class ActivationFunction {
    protected final int featureSize;
    protected final int neurons;

    ActivationFunction(int featureSize, int neurons) {
        this.featureSize = featureSize;
        this.neurons = neurons;
    }

    abstract Vector out(Vector z);

    public abstract Vector derivativeByZ(Vector z, int aPos);

    Vector[] getW() {
        Vector[] W = new Vector[neurons];

        for (int i = 0; i < W.length; i++) {
            W[i] = new Vector(featureSize);
            W[i].randomise(1.0);
        }

        return W;
    }

    Vector getB() {
        Vector b = new Vector(neurons);
        b.randomise(1.0);

        return b;
    }

    Vector z(Vector[] W, Vector B, Vector v) {
        double[] points = new double[neurons];

        for (int i = 0; i < neurons; i++) {
            points[i] = W[i].dot(v) + B.x(i);
        }

        return new Vector(points);
    }
}
