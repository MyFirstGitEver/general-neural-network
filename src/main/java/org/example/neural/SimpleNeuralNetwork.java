package org.example.neural;

import org.example.Pair;
import org.example.Vector;
import org.example.others.DataGetter;

import java.io.*;

public class SimpleNeuralNetwork {

    public interface Loss {
        double loss(Vector v, Vector y) throws Exception;
        Vector derivativeByA(Vector A, Vector Y);
    }

    private final DenseLayer[] layers;
    private final Loss loss;
    private final DataGetter<Vector> xGetter;
    private final DataGetter<Vector> yGetter;
    private final static double beta = 0.9f;
    private final static double beta2 = 0.999f;

    public SimpleNeuralNetwork(DenseLayer[] layers, Loss loss, DataGetter<Vector> xGetter, DataGetter<Vector> yGetter) {
        this.layers = layers;
        this.loss = loss;

        this.xGetter = xGetter;
        this.yGetter = yGetter;
    }

    public SimpleNeuralNetwork(DenseLayer[] layers, String alias) throws IOException {
        this.layers = layers;
        loss = null;
        xGetter = null;
        yGetter = null;

        loadParams(alias);
    }

    public Vector predict(Vector v) {
        Vector curr = v;

        for (DenseLayer layer : layers) {
            curr = layer.output(curr).second;
        }

        return curr;
    }

    public void train(double learningRate,
                      int iter,
                      int batchSize,
                      int maxToSave,
                      String alias, boolean printCost, boolean printLog) throws Exception {
        BufferedWriter writer = null;

        if(printLog) {
            writer = new BufferedWriter(new FileWriter("./log.txt"));
        }

        int iteration = 0;

        // dW and dB holds current gradient update
        Matrix[] dW = new Matrix[layers.length];
        Vector[] dB = new Vector[layers.length];

        Matrix[] firstMomentW = new Matrix[layers.length];
        Matrix[] secondMomentW = new Matrix[layers.length];
        Vector[] firstMomentB = new Vector[layers.length];
        Vector[] secondMomentB = new Vector[layers.length];

        for(int i=0;i<layers.length;i++) {
            Pair<Integer, Integer> wShape = layers[i].wShape();

            dW[i] = new Matrix(wShape.first, wShape.second);
            dB[i] = new Vector(layers[i].bShape());

            firstMomentW[i] = new Matrix(wShape.first, wShape.second);
            secondMomentW[i] = new Matrix(wShape.first, wShape.second);

            firstMomentB[i] = new Vector(layers[i].bShape());
            secondMomentB[i] = new Vector(layers[i].bShape());
        }

        while(iteration < iter) {
            if(iteration % maxToSave == 0) {
                saveParams(alias);
                System.out.print(iteration + " iterations have passed.");
                if(printCost) {
                    System.out.print(" Cost: " + cost());
                }

                System.out.println();
            }

            for(int j=0;j<xGetter.size();j+= Math.min(batchSize, xGetter.size() - j)) {
                for (Matrix matrix : dW) {
                    matrix.reset();
                }

                for(Vector v : dB) {
                    v.reset();
                }

                computeGradient(j, j + Math.min(batchSize - 1, xGetter.size() - 1 - j), dW, dB);

                for(Matrix matrix : dW) {
                    matrix.normalizeByRow();
                }

                for(Vector v : dB) {
                    v.normalise();
                }

                log(dW, dB, writer);

                for(int i=0;i<layers.length;i++) {
                    dW[i].divideBy(xGetter.size());
                    dB[i].divideBy(xGetter.size());

                    firstMomentW[i].scale(beta).add(dW[i].copy().scale(1 - beta));
                    secondMomentW[i].scale(beta2).add(dW[i].square().scale(1 - beta2));

                    firstMomentB[i].scaleBy(beta).add(dB[i].copy().scaleBy(1 - beta));
                    secondMomentB[i].scaleBy(beta2).add(dB[i].square().scaleBy(1 - beta2));

                    layers[i].update(
                            firstMomentW[i].vectorize(true),
                            secondMomentW[i].vectorize(true),
                            firstMomentB[i],
                            secondMomentB[i], learningRate);
                }
            }

            iteration++;
        }

        if(writer != null) {
            writer.close();
        }
        saveParams(alias);
    }

    public double cost() throws Exception {
        double total = 0;

        for(int i=0;i<xGetter.size();i++) {
            total += loss.loss(predict(xGetter.at(i)), yGetter.at(i));
        }

        return total / xGetter.size();
    }

    public Vector[] w(int i) {
        return layers[i].getW();
    }

    public Vector b(int i) {
        return layers[i].getB();
    }

    private void log(Matrix[] dW, Vector[] dB, BufferedWriter writer) throws IOException {
        if(writer == null) {
            return;
        }

        for(int i=dB.length-1;i>=0;i--) {
            writer.write(dB[i].toString());
            writer.write("\2");
        }

        writer.write("\3");

        for(int i=dW.length-1;i>=0;i--) {
            writer.write(dW[i].toString());
            writer.write("\4");
        }

        writer.newLine();
    }

    private Vector lastError(Pair<Vector, Vector> lastZAndA, Vector y) {
        Vector z = lastZAndA.first;
        Vector a = lastZAndA.second;

        return loss.derivativeByA(a, y).hadamard(layers[layers.length - 1].derivativeByZ(z, y));
    }

    private void computeGradient(int from, int to, Matrix[] dW, Vector[] dB) throws Exception {
        for(int iteration=from;iteration<=to;iteration++) {
            Pair<Vector, Vector> point = new Pair<>(xGetter.at(iteration), yGetter.at(iteration));

            Pair<Vector, Vector>[] zAndA = new Pair[layers.length];

            Vector curr = point.first;
            // Forward propagation
            for (int i=0;i<layers.length;i++) {
                zAndA[i] = layers[i].output(curr);
                curr = zAndA[i].second;
            }

            Vector currError = lastError(zAndA[zAndA.length - 1], point.second);

            for(int i=zAndA.length - 1;i>=0;i--) {
                Vector a;

                if(i == 0) {
                    a = point.first;
                }
                else {
                    a = zAndA[i - 1].second;
                }

                Matrix matError = new Matrix(currError, true);
                Vector[] gradientW = matError.mul(new Matrix(a, false));

                Vector nextError = null;
                if(i != 0) {
                    // W^T . lastError * g'(z)
                    // only makes sense when we're not finished
                    nextError = new Vector(layers[i].transposeOfW().mul(matError))
                            .hadamard(layers[i - 1].derivativeByZ(zAndA[i - 1].first, point.second));
                }

                dW[i].add(gradientW);
                dB[i].add(currError);

                currError = nextError;
            }
        }
    }

    public void saveParams(String alias) throws IOException {
        File f = new File("layers" + alias);
        f.mkdir();

        for(int i=0;i<layers.length;i++) {
            File dir = new File(f.getAbsolutePath() + "\\layer " + (i + 1));
            dir.mkdir();

            BufferedWriter wWriter = new BufferedWriter(new FileWriter(dir.getAbsolutePath() + "\\w"));
            BufferedWriter bWriter = new BufferedWriter(new FileWriter(dir.getAbsolutePath() + "\\b"));

            Vector[] w = layers[i].getW();

            for(Vector v : w) {
                wWriter.write(v.toString());
                wWriter.newLine();
            }

            Vector b = layers[i].getB();
            bWriter.write(b.toString());

            wWriter.close();
            bWriter.close();
        }
    }

    public boolean loadParams(String alias) throws IOException {
        File f = new File("layers" + alias);

        if(!f.exists()) {
            return false;
        }

        int layerNum = f.listFiles().length;
        for(int i=0;i<layerNum;i++) {
            File dir = new File(f, "layer " + (i + 1));

            BufferedReader wReader = new BufferedReader(new FileReader(dir.getAbsolutePath() + "\\w"));
            BufferedReader bReader = new BufferedReader(new FileReader(dir.getAbsolutePath() + "\\b"));

            Vector[] w = layers[i].getW();

            String line;

            int index = 0;
            while((line = wReader.readLine()) != null) {
                w[index] = new Vector(line);
                index++;
            }

            layers[i].setB(new Vector(bReader.readLine()));

            wReader.close();
            bReader.close();
        }

        return true;
    }
}