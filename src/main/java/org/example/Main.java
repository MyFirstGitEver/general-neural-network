package org.example;

import org.example.activators.Activator;
import org.example.activators.FeedForwardSoftmaxActivator;
import org.example.activators.FeedforwardReluActivator;
import org.example.loss.CrossEntropy;
import org.example.networks.FeedforwardNetwork;
import org.example.neural.DenseLayer;
import org.example.neural.ReluActivation;
import org.example.neural.SimpleNeuralNetwork;
import org.example.neural.SoftMaxActivation;
import org.example.others.DataGetter;

public class Main {
    public static void main(String[] args) throws Exception {
        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\train.xlsx");
        Pair<Vector, Vector>[] dataset = reader.createLabeledDataset(0, 0, 0);
        reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\test.xlsx");
        Pair<Vector, Vector>[] testSet = reader.createLabeledDataset(0, 0, 0);

        graphTest(dataset, testSet);
    }

    private void neuralTest() throws Exception {
        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\train.xlsx");
        Pair<Vector, Vector>[] dataset = reader.createLabeledDataset(0, 0, 0);
        reader = new ExcelReader("D:\\Source code\\Data\\Titanic\\test.xlsx");
        Pair<Vector, Vector>[] testSet = reader.createLabeledDataset(0, 0, 0);

        DataGetter<Vector> xGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return dataset[i].first;
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return dataset[i].second;
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(
                new DenseLayer[] {
                        new DenseLayer(new ReluActivation(dataset[0].first.size() , 3)),
                        new DenseLayer(new SoftMaxActivation(3, 2))
                }, new org.example.neural.CrossEntropy(), xGetter, yGetter);

        model.train(0.001, 300, 150, 5, "", true, false);

        System.out.println("Cost of training set is: " + model.cost());

        int hit = 0;

        for (Pair<Vector, Vector> vectorVectorPair : testSet) {
            Vector confidence = model.predict(vectorVectorPair.first);

            if (confidence.x(0) >= confidence.x(1) && vectorVectorPair.second.x(0) == 1) {
                hit++;
            } else if (confidence.x(0) < confidence.x(1) && vectorVectorPair.second.x(1) == 1) {
                hit++;
            }
        }

        System.out.println("Accuracy reached: " + (double)hit / testSet.length * 100 + " %");
    }

    private static void graphTest(Pair<Vector, Vector>[] dataset, Pair<Vector, Vector>[] testSet) throws Exception {
        DataGetter<Vector> xGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return dataset[i].first;
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return dataset[i].second;
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        FeedforwardNetwork model = new FeedforwardNetwork(
                new int[] { dataset[0].first.size(), 3, 2},
                new Activator[] {
                        new FeedforwardReluActivator(),
                        new FeedForwardSoftmaxActivator(),
                }, new CrossEntropy(), xGetter, yGetter);

        model.train(0.01, 300, 150, null, null,true);

        System.out.println("Cost of training set is: " + model.cost());

        int hit = 0;

        for (Pair<Vector, Vector> vectorVectorPair : testSet) {
            Vector confidence = model.output(vectorVectorPair.first);

            if (confidence.x(0) >= confidence.x(1) && vectorVectorPair.second.x(0) == 1) {
                hit++;
            } else if (confidence.x(0) < confidence.x(1) && vectorVectorPair.second.x(1) == 1) {
                hit++;
            }
        }

        System.out.println("Accuracy reached: " + (double)hit / testSet.length * 100 + " %");
    }

    static void normalise(Pair<Vector, Vector>[] xTrain, Pair<Vector, Vector>[] xTest) {
        double[] mean = new double[xTrain[0].first.size()];
        double[] std = new double[xTrain[0].first.size()];

        for(Pair<Vector, Vector> p : xTrain) {
            for(int i=0;i<p.first.size();i++) {
                mean[i] += p.first.x(i);
            }
        }

        for(int i=0;i<mean.length;i++) {
            mean[i] /= xTrain.length;
        }

        for(Pair<Vector, Vector> p : xTrain) {
            for(int i=0;i<p.first.size();i++) {
                double term = (p.first.x(i) - mean[i]);
                std[i] += term * term;
            }
        }

        for(int i=0;i<std.length;i++) {
            std[i] = Math.sqrt(std[i] / xTrain.length);
        }

        // Normalise
        for(Pair<Vector, Vector> p : xTrain) {
            for(int i=0;i<std.length;i++) {
                p.first.setX(i, (p.first.x(i) - mean[i]) / std[i]);
            }
        }

        if(xTest == null) {
            return;
        }

        for(Pair<Vector, Vector> p : xTest) {
            for(int i=0;i<std.length;i++) {
                p.first.setX(i, (p.first.x(i) - mean[i]) / std[i]);
            }
        }
    }
}