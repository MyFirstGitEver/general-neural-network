package org.example;

import org.example.chooser.numbergenerators.BackwardGenerator;
import org.example.neural.*;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

public class RNNTester {
    private final TanhActivation tanhActivation;
    private final SoftMaxActivation softMaxActivation;

    private final SimpleNeuralNetwork.Loss loss;

    private final int priorKnowledgeSize;

    private final Matrix Whx, Whh, Wyh;
    private final Vector bh, by;

    private final Matrix dWhx, dWhh, dWyh;
    private final Vector dbh, dby;

    private final int timeStep;

    public RNNTester(int inputSize, int outputSize, int priorKnowledgeSize, int timeStep) {
        this.priorKnowledgeSize = priorKnowledgeSize;

        double scaleFactor = 3.0;

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

        this.timeStep = timeStep;
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
        Vector[] ZH = new Vector[input.length];

        //feedforward
        H[0] = new Vector(priorKnowledgeSize);
        for(int i=0;i<input.length;i++) {
            Vector v = input[i];

            Vector[] term1 = Whx.mul(new Matrix(v, true));
            Vector[] term2 = Whh.mul(new Matrix(H[i], true));

            ZH[i] = new Vector(term1).add(new Vector(term2)).add(bh);
            H[i + 1] = tanhActivation.out(ZH[i]);

            ZY[i] = new Vector(Wyh.mul(new Matrix(H[i + 1], true))).add(by);
            Y[i] = softMaxActivation.out(ZY[i]);
        }

        dW(label, Y, H, ZY, ZH, input, input.length);
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

    private void dW(Vector label,
                    Vector[] Y,
                    Vector[] H,
                    Vector[] ZY,
                    Vector[] ZH,
                    Vector[] input, int inputLength) throws Exception {
        for(int source=inputLength - 1;source>=0;source--) {
            Vector dz = error(ZY[source], loss.derivativeByA(Y[source], label), softMaxActivation);

            Matrix matError = new Matrix(dz, true);
            Vector[] gradientWyh = matError.mul(new Matrix(H[source + 1], false));
            dWyh.add(gradientWyh);
            dby.add(dz);

            Vector nextError = new Vector(Matrix.transpose(Wyh.vectorize(true)).mul(matError));

            for(int i=source;i>=0;i--) {
                nextError = error(ZH[i],nextError, tanhActivation);

                matError = new Matrix(nextError, true);
                Vector[] gradientWhh = matError.mul(new Matrix(H[source], false));
                dWhh.add(gradientWhh);

                Vector[] gradientWhx = matError.mul(new Matrix(input[source], false));
                dWhx.add(gradientWhx);

                dbh.add(nextError);

                nextError = new Vector(Matrix.transpose(Whh.vectorize(true)).mul(matError));
            }
        }

        log();
    }

    private void log() throws Exception {
        List<TestingObject> tests = new ArrayList<>(1);

        tests.add(new TestingObject(new Matrix[] {
                dWyh,
                dWhh.concatToLeft(dWhx)
        }, new Vector[] {
                dby,
                dbh
        }, new int[] { 0,  by.size()}, new BackwardGenerator(1, -1)));

        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("./log.txt"));
        oos.writeObject(tests);
        oos.close();
    }

    private Vector error(Vector z, Vector dLdA, ActivationFunction activationFunction) {
        Vector result = new Vector(z.size());

        for(int i=0;i<result.size();i++) {
            result.setX(i, dLdA.hadamard(activationFunction.derivativeByZ(z, i)).sum());
        }

        return result;
    }
}