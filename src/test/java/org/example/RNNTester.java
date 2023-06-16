package org.example;

import org.example.neural.*;

public class RNNTester {
    private final TanhActivation tanhActivation;
    private final SoftMaxActivation softMaxActivation;

    private final SimpleNeuralNetwork.Loss loss;

    private final int priorKnowledgeSize;

    private final Matrix Whx, Whh, Wyh;
    private final Vector bh, by;

    private final Matrix dWhx, dWhh, dWyh;
    private final Vector dbh, dby;

    private final int layers;

    public RNNTester(int inputSize, int outputSize, int priorKnowledgeSize, int layers) {
        this.priorKnowledgeSize = priorKnowledgeSize;

        double scaleFactor = 0.1;

        Whx = new Matrix(priorKnowledgeSize, inputSize);
        dWhx = new Matrix(priorKnowledgeSize, inputSize);
        Whx.randomise(scaleFactor);

        Whh = new Matrix(priorKnowledgeSize, priorKnowledgeSize);
        dWhh = new Matrix(priorKnowledgeSize, priorKnowledgeSize);
        Whh.randomise(scaleFactor);

        bh = new Vector(priorKnowledgeSize);
        dbh = new Vector(priorKnowledgeSize);
        bh.randomise(scaleFactor);

        Wyh = new Matrix(outputSize, priorKnowledgeSize);
        dWyh = new Matrix(outputSize, priorKnowledgeSize);
        Wyh.randomise(scaleFactor);

        by = new Vector(outputSize);
        dby = new Vector(outputSize);
        by.randomise(scaleFactor);

        tanhActivation = new TanhActivation(inputSize, priorKnowledgeSize);
        softMaxActivation = new SoftMaxActivation(priorKnowledgeSize, outputSize);
        loss = new CrossEntropy();

        this.layers = layers;
    }

    public Matrix getWhx() {
        return Whx;
    }

    public Matrix getWhh() {
        return Whh;
    }

    public Matrix getWyh() {
        return Wyh;
    }

    public Vector getBh() {
        return bh;
    }

    public Vector getBy() {
        return by;
    }

    public Vector[] feedforward(Vector[] input) throws Exception {
        Vector[] y = new Vector[input.length];
        Vector lastH = new Vector(priorKnowledgeSize);

        for(int i=0;i<input.length;i++) {
            Vector v = input[i];

            Vector[] term1 = Whx.mul(new Matrix(v, true));
            Vector[] term2 = Whh.mul(new Matrix(lastH, true));

            Vector h = new Vector(term1).add(new Vector(term2)).add(bh);
            h = tanhActivation.out(h);

            y[i] = softMaxActivation.out(new Vector(Wyh.mul(new Matrix(h, true))).add(by));
            lastH = h;
        }

        return y;
    }

    public void backward(Vector[] input, Vector label) throws Exception {
        Vector[] Y = new Vector[input.length];
        Vector[] H = new Vector[input.length + 1];
        Vector[] ZY = new Vector[input.length];

        //feedforward
        H[0] = new Vector(priorKnowledgeSize);
        for(int i=0;i<input.length;i++) {
            Vector v = input[i];

            Vector[] term1 = Whx.mul(new Matrix(v, true));
            Vector[] term2 = Whh.mul(new Matrix(H[i], true));

            Vector zh = new Vector(term1).add(new Vector(term2)).add(bh);
            H[i + 1] = tanhActivation.out(zh);

            ZY[i] = new Vector(Wyh.mul(new Matrix(H[i + 1], true))).add(by);
            Y[i] = softMaxActivation.out(ZY[i]);
        }

        dWyh(label, Y, H, ZY, input.length);
    }

    public Matrix getdWhx() {
        return dWhx;
    }

    public Matrix getdWhh() {
        return dWhh;
    }

    public Matrix getdWyh() {
        return dWyh;
    }

    public Vector getDbh() {
        return dbh;
    }

    public Vector getDby() {
        return dby;
    }

    private void dWyh(Vector label, Vector[] Y, Vector[] H, Vector[] ZY, int inputLength) throws Exception {
        for(int i=0;i<inputLength;i++) {
            Vector error =
                    loss.derivativeByA(Y[i], label).hadamard(softMaxActivation.derivativeByZ(ZY[i], label));

            Matrix matError = new Matrix(error, true);
            Vector[] gradientW = matError.mul(new Matrix(H[i + 1], false));

            dWyh.add(gradientW);
        }

        dWyh.normalizeByRow();
    }
}