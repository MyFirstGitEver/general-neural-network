package org.example.neural;

import org.example.Vector;

public class SoftMaxActivation extends ActivationFunction {
    public SoftMaxActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    public Vector out(Vector z) {
        double total = 0;
        double max = -Double.MAX_VALUE;

        for(int i=0;i<z.size();i++) {
            max = Math.max(max, z.x(i));
        }

        for(int i=0;i<z.size();i++) {
            total += Math.exp(z.x(i) - max);
        }

        Vector a = new Vector(z.size());
        for(int i=0;i<z.size();i++) {
            a.setX(i, (Math.exp(z.x(i) - max) / total));
        }

        return a;
    }

    @Override
    public Vector derivativeByZ(Vector z, int zIndex) {
        Vector a = out(z);
        Vector dz = new Vector(a.size());

        for(int i=0;i<dz.size();i++) {
            if(i == zIndex) {
                dz.setX(i, a.x(i) * (1 - a.x(i)));
            }
            else {
                dz.setX(i, - a.x(i) * a.x(zIndex));
            }
        }

        return dz;
    }
}