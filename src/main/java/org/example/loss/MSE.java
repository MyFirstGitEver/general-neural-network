package org.example.loss;

import org.example.Vector;

public class MSE implements Loss {
    @Override
    public double loss(Vector A, Vector Y) {
        double loss = 0;

        for(int i=0;i<A.size();i++) {
            double term = A.x(i) - Y.x(i);

            loss += (term * term) / 2;
        }

        return loss;
    }

    @Override
    public Vector dA(Vector A, Vector Y) {
        Vector delta = new Vector(A.size());

        for(int i=0;i<A.size();i++) {
            delta.setX(i, A.x(i) - Y.x(i));
        }

        return delta;
    }
}