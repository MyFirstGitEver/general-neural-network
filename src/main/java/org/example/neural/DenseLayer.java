package org.example.neural;

import org.example.Pair;
import org.example.Vector;

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
            W[i].subtract(firstMomentW[i].divideCopy(secondMomentW[i].sqrtCopy(), 10e-8f).scaleBy(learningRate));
        }

        for(int i=0;i<firstMomentB.size();i++) {
            B.subtract(firstMomentB.divideCopy(secondMomentB.sqrtCopy(), 10e-8f).scaleBy(learningRate));
        }
    }

    public Vector derivativeByZ(Vector Z, int zIndex) {
        return function.derivativeByZ(Z, zIndex);
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