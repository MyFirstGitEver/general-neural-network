package org.example.neural;

import org.example.Vector;

public class MSE implements SimpleNeuralNetwork.Loss {
    @Override
    public double loss(Vector v, Vector y) {
        double loss = 0;

        for(int i=0;i<v.size();i++) {
            double term = v.x(i) - y.x(i);

            loss += (term * term) / 2;
        }

        return loss;
    }

    @Override
    public Vector derivativeByA(Vector A, Vector Y) {
        Vector v = new Vector(A.size());

        for(int i=0;i<v.size();i++) {
            v.setX(i, (A.x(i) - Y.x(i)));
        }

        return v;
    }
}
