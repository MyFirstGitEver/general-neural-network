package org.example.loss;

import org.example.Vector;

public class CrossEntropy implements Loss {
    @Override
    public double loss(Vector a, Vector y) throws Exception {
        for(int i=0;i<y.size();i++) {
            if(y.x(i) == 1) {
                return -Math.log(a.x(i) + 0.00001);
            }
        }

        throw new Exception("Label has some problem with it!");
    }

    @Override
    public Vector dA(Vector A, Vector Y) {
        Vector delta = new Vector(A.size());

        for(int i=0;i<A.size();i++) {
            delta.setX(i, - Y.x(i) / (A.x(i) + 1e-8));
        }

        return delta;
    }
}