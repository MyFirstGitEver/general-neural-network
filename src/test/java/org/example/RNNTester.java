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

    private final int priorKnowledgeSize, layers;

    private final Matrix Whx, Whh, Wyh;

    private final Vector bh, by;

    private final Matrix dWhx, dWhh, dWyh;
    private final Vector dbh, dby;

    private final Matrix firstWhx, firstWhh, firstWyh;

    private final Matrix secondWhx, secondWhh, secondWyh;

    private final Vector firstBh, firstBy;

    private final Vector secondBh, secondBy;

    private final List<TestingObject> tests = new ArrayList<>();

    private final List<TestingObject> updatedParams = new ArrayList<>();

    private final static double beta = 0.9f;
    private final static double beta2 = 0.999f;

    public RNNTester(int inputSize, int outputSize, int priorKnowledgeSize, int layers) {
        this.priorKnowledgeSize = priorKnowledgeSize;
        this.layers = layers;

        double scaleFactor = 3.0;

        Whx = new Matrix(priorKnowledgeSize, inputSize);
        dWhx = new Matrix(priorKnowledgeSize, inputSize);
        firstWhx = new Matrix(priorKnowledgeSize, inputSize);
        secondWhx = new Matrix(priorKnowledgeSize, inputSize);
        Whx.randomise(scaleFactor);

        Whh = new Matrix(priorKnowledgeSize, priorKnowledgeSize);
        dWhh = new Matrix(priorKnowledgeSize, priorKnowledgeSize);
        firstWhh = new Matrix(priorKnowledgeSize, priorKnowledgeSize);
        secondWhh = new Matrix(priorKnowledgeSize, priorKnowledgeSize);
        Whh.randomise(scaleFactor);

        Wyh = new Matrix(outputSize, priorKnowledgeSize);
        dWyh = new Matrix(outputSize, priorKnowledgeSize);
        firstWyh = new Matrix(outputSize, priorKnowledgeSize);
        secondWyh = new Matrix(outputSize, priorKnowledgeSize);
        Wyh.randomise(scaleFactor);

        bh = new Vector(priorKnowledgeSize);
        dbh = new Vector(priorKnowledgeSize);
        firstBh = new Vector(priorKnowledgeSize);
        secondBh = new Vector(priorKnowledgeSize);
        bh.randomise(scaleFactor);

        by = new Vector(outputSize);
        dby = new Vector(outputSize);
        firstBy = new Vector(outputSize);
        secondBy = new Vector(outputSize);
        by.randomise(scaleFactor);

        tanhActivation = new TanhActivation(inputSize, priorKnowledgeSize);
        softMaxActivation = new SoftMaxActivation(priorKnowledgeSize, outputSize);
        loss = new CrossEntropy();
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

    public void train(Vector[][] dataset, Vector[] labels, int batchSize, int trainCnt, double learningRate) throws Exception {
        for (int trainCount = 0; trainCount < trainCnt; trainCount++) {
            for (int i = 0; i < dataset.length; i += batchSize) {
                int from = i;
                int to = i + Math.min(batchSize, dataset.length - i) - 1;

                dWyh.reset();
                dWhh.reset();
                dWhx.reset();
                dby.reset();
                dbh.reset();

                for (int iteration = from; iteration <= to; iteration++) {
                    backward(dataset[iteration], labels[iteration]);
                }

                addDeltaToLog();

                int currentBatchSize = to - from + 1;

                dWyh.divideBy(layers * currentBatchSize);
                dWhh.divideBy(layers * currentBatchSize);
                dWhx.divideBy(layers * currentBatchSize);

                dby.divideBy(layers * currentBatchSize);
                dbh.divideBy(layers * currentBatchSize);

                updateW(Wyh, firstWyh, secondWyh, dWyh, learningRate);
                updateW(Whh, firstWhh, secondWhh, dWhh, learningRate);
                updateW(Whx, firstWhx, secondWhx, dWhx, learningRate);

                updateB(by, firstBy, secondBy, dby, learningRate);
                updateB(bh, firstBh, secondBh, dbh, learningRate);
                
                addParamsToLog();
            }
        }

        // write log to file
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("./log.txt"));
        oos.writeObject(tests);
        oos.close();

        oos = new ObjectOutputStream(new FileOutputStream("./params.txt"));
        oos.writeObject(updatedParams);
        oos.close();
    }

    private void updateW(Matrix W, Matrix firstMoment, Matrix secondMoment, Matrix dW, double learningRate) throws Exception {
        firstMoment.scale(beta).add(dW.copy().scale(1 - beta));
        secondMoment.scale(beta2).add(dW.square().scale(1 - beta2));

        W.selfSubtract(firstMoment.hadamardDivideCopy(secondMoment.sqrtCopy(), 10e-8f).scale(learningRate));
    }

    private void updateB(Vector B, Vector firstMoment, Vector secondMoment, Vector dB, double learningRate) throws Exception {
        firstMoment.scaleBy(beta).add(dB.copy().scaleBy(1 - beta));
        secondMoment.scaleBy(beta2).add(dB.square().scaleBy(1 - beta2));

        B.subtract(firstMoment.divideCopy(secondMoment.sqrtCopy(), 10e-8f).scaleBy(learningRate));
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
    }

    private void addDeltaToLog() throws Exception {
        Matrix dWyhCopy = dWyh.copy();
        Vector dbyCopy = dby.copy();
        Vector dbhCopy = dbh.copy();

        tests.add(new TestingObject(new Matrix[] {
                dWyhCopy,
                dWhh.concatToLeftCopy(dWhx)
        }, new Vector[] {
                dbyCopy,
                dbhCopy
        }, new int[] { 0,  by.size()}, new BackwardGenerator(1, -1)));
    }

    private void addParamsToLog() {
        Matrix[] W = new Matrix[3];
        Vector[] B = new Vector[3];

        W[0] = this.Wyh.copy();
        W[1] = this.Whh.copy();
        W[2] = this.Whx.copy();

        B[0] = this.by.copy();
        B[1] = this.bh.copy();

        updatedParams.add(new TestingObject(W, B, new int[] { 0,  by.size()}, new BackwardGenerator(1, - 1)));
    }

    private Vector error(Vector z, Vector dLdA, ActivationFunction activationFunction) {
        Vector result = new Vector(z.size());

        for(int i=0;i<result.size();i++) {
            result.setX(i, dLdA.hadamard(activationFunction.derivativeByZ(z, i)).sum());
        }

        return result;
    }
}