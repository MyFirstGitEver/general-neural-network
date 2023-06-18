package org.example;

import org.example.activators.Activator;
import org.example.activators.FeedForwardLinearActivator;
import org.example.activators.FeedforwardReluActivator;
import org.example.loss.MSE;
import org.example.networks.FeedforwardNetwork;
import org.example.neural.DenseLayer;
import org.example.neural.LinearActivation;
import org.example.neural.ReluActivation;
import org.example.neural.SimpleNeuralNetwork;
import org.example.others.DataGetter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class GradientTesting {
    public static Stream<Arguments> cases2() {
        return Stream.of(oneHundredArgs());
    }

    private static Arguments[] oneHundredArgs() {
        Arguments[] args = new Arguments[20];

        for(int i=0;i<args.length;i++) {
            double learningRate = Math.random();
            int iteration = (int) (20 + Math.random() * 150);
            int datasetSize = (int) (3000 + 6000 * Math.random());
            int batchSize = (int) (Math.random() * 70 + 30);

            args[i] = Arguments.of(
                    learningRate, 20 + iteration, Math.min(datasetSize, batchSize), datasetSize);
        }

        return args;
    }

    @ParameterizedTest
    @MethodSource("cases2")
    public void backwardTesting(double learningRate, int iteration, int batchSize, int datasetSize) throws Exception {
        Pair<DataGetter<Vector>, DataGetter<Vector>> result = fetchData(datasetSize);
        int featureSize = result.first.at(0).size();

        FeedforwardNetwork network = new FeedforwardNetwork(new int[] {
                featureSize, 15, 3, 1
        }, new Activator[] {
                new FeedforwardReluActivator(),
                new FeedforwardReluActivator(),
                new FeedForwardLinearActivator()
        }, new MSE(), result.first, result.second);

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(new DenseLayer[]{
                new DenseLayer(new ReluActivation(featureSize, 15)),
                new DenseLayer(new ReluActivation(15, 3)),
                new DenseLayer(new LinearActivation(3, 1))
        }, new org.example.neural.MSE(), result.first, result.second);

        network.loadTestingToLayer(model.w(0), model.b(0), 0);
        network.loadTestingToLayer(model.w(1), model.b(1), 1);
        network.loadTestingToLayer(model.w(2), model.b(2), 2);

        model.train(learningRate, iteration, batchSize, 5, "", false, true);
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./log.txt"));
        ObjectInputStream paramsOis = new ObjectInputStream(new FileInputStream("./params.txt"));

        boolean passed = network.train(learningRate, iteration, batchSize, (List<TestingObject>) ois.readObject(),
                (List<TestingObject>) paramsOis.readObject(), false);
        ois.close();
        paramsOis.close();

        //Assertions.assertTrue(Math.abs(model.cost() - network.cost()) < 5.0);
        Assertions.assertTrue(passed);
    }

    private Pair<DataGetter<Vector>, DataGetter<org.example.Vector>> fetchData(int datasetSize) throws Exception {
        ExcelReader reader = new ExcelReader("D:\\Source code\\Data\\Plant\\plant.xlsx");

        Pair<Vector, Vector>[] dataset = new Pair[reader.getRowCount() - 1]; // exclude the first row

        for (int i = 1; i <= dataset.length; i++) {
            Object[] data = reader.getRow(i, 0);
            Object[] x = Arrays.copyOfRange(data, 0, data.length - 1);

            dataset[i - 1] = new Pair<>(new Vector(x), new Vector(((Double) data[data.length - 1]).doubleValue()));
        }

        Pair<Vector, Vector>[] xTrain = Arrays.copyOfRange(dataset, 0, datasetSize);
//        Pair<Vector, Vector>[] xTest = Arrays.copyOfRange(dataset,
//                (int) (0.8 * dataset.length), dataset.length);

        DataGetter<Vector> xGetter = new DataGetter<>() {
            @Override
            public Vector at(int i) {
                return xTrain[i].first;
            }

            @Override
            public int size() {
                return xTrain.length;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return xTrain[i].second;
            }

            @Override
            public int size() {
                return xTrain.length;
            }
        };

        Main.normalise(xTrain,null);
        return new Pair<>(xGetter, yGetter);
    }
}