package org.example.neural;

import org.example.Pair;
import org.example.Vector;

abstract class ActivationFunction {
    protected final int featureSize;
    protected final int neurons;

    ActivationFunction(int featureSize, int neurons) {
        this.featureSize = featureSize;
        this.neurons = neurons;
    }

    abstract Vector out(Vector z);
    abstract Vector derivativeByZ(Vector z, Vector y);

    Vector[] getW() {
        Vector[] W = new Vector[neurons];

        for(int i=0;i<W.length;i++) {
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

        for(int i=0;i<neurons;i++) {
            points[i] = W[i].dot(v) + B.x(i);
        }

        return new Vector(points);
    }
}

public class DenseLayer {
    private final ActivationFunction function;

    private final Vector[] W;
    private Vector B;

    public DenseLayer(ActivationFunction function) {
        this.function = function;

        W = function.getW();
        B = function.getB();
    }

    public Pair<Vector, Vector> output(Vector v) {
        if(v.size() != W[0].size()) {
            throw new RuntimeException("Wrong input dimension!");
        }

        // Calculate z for activation function
        Vector z = function.z(W, B, v);

        return new Pair<>(z, function.out(z));
    }

    public Vector[] getW() {
        return W;
    }

    public Vector getB() {
        return B;
    }

    public void setB(Vector B) {
        this.B = B;
    }

    public void update(
            Vector[] firstMomentW,
            Vector[] secondMomentW,
            Vector firstMomentB,
            Vector secondMomentB,
            double learningRate)
            throws Exception {

        for(int i=0;i<firstMomentW.length;i++) {
            W[i].subtract(firstMomentW[i].divide(secondMomentW[i].sqrt(), 10e-8f).scaleBy(learningRate));
        }

        for(int i=0;i<firstMomentB.size();i++) {
            B.subtract(firstMomentB.divide(secondMomentB.sqrt(), 10e-8f).scaleBy(learningRate));
        }
    }

    public Vector derivativeByZ(Vector Z, Vector y) {
        return function.derivativeByZ(Z, y);
    }

    public Matrix transposeOfW() {
        return Matrix.transpose(W);
    }

    public Pair<Integer, Integer> wShape() {
        return new Pair<>(W.length, W[0].size());
    }

    public int bShape() {
        return B.size();
    }
}