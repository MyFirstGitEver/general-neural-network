package org.example;

import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.neural.Matrix;

import java.io.Serializable;

public class TestingObject implements Serializable {
    private final Matrix[] W;
    private final Vector[] B;

    private final int[] offset;

    private final NumberGenerator layerChooser;

    public TestingObject(Matrix[] w, Vector[] b, int[] offset, NumberGenerator layerChooser) {
        W = w;
        B = b;
        this.offset = offset;
        this.layerChooser = layerChooser;
    }

    public Matrix[] getdW() {
        return W;
    }

    public Vector[] getdB() {
        return B;
    }

    public int[] getOffset() {
        return offset;
    }

    public NumberGenerator getLayerChooser() {
        return layerChooser;
    }
}