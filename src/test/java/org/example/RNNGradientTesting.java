package org.example;

import org.example.networks.RNN;
import org.example.others.DataGetter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import java.util.List;
import java.util.stream.Stream;

public class RNNGradientTesting {
    public static Stream<Arguments> cases2() {
        return Stream.of(args());
    }

    public static Vector[] vectors(int num, Integer size) {
        Vector[] set = new Vector[num];

        for(int j=0;j<num;j++) {
            if(size == null) {
                size = (int) (Math.random() * 2 + 3);
            }

            Vector v = new Vector(size);
            set[j] = v;

            for(int i=0;i<v.size();i++) {
                v.setX(i, Math.random() * 0.01);
            }
        }

        return set;
    }

    private static Arguments[] args() {
        Arguments[] args = new Arguments[40];

        for(int i=0;i<args.length;i++) {
            int iteration = (int) (3  + Math.random() * 7);
            int datasetSize = (int) (Math.random() * 100 + 200);
            int inputSize = (int) (Math.random() * 10 + 315);
            int outputSize = (int) (Math.random() * 2 + 2);

            Vector[][] dataset = new Vector[datasetSize][];
            Vector[] labels = new Vector[datasetSize];

            int timeStep = (int) (Math.random() * 10) + 1;

            for(int j=0;j<dataset.length;j++) {
                dataset[j] = vectors(timeStep, inputSize);
            }

            for(int j=0;j<dataset.length;j++) {
                labels[j] = randomOneHotVector(outputSize);
            }

            args[i] = Arguments.of(
                    inputSize,
                    outputSize,
                    (int) (Math.random() * 15 + 30),
                    timeStep, dataset, labels, (int) Math.min(dataset.length, Math.random() * 70 + 30), iteration);
        }


        return args;
    }

    private static Vector randomOneHotVector(int size) {
        Vector y = new Vector(size);
        int spot = (int) (Math.random() * (size));
        y.setX(spot, 1);

        return y;
    }

    @ParameterizedTest
    @MethodSource("cases2")
    public void testWyhGradient(int inputSize, int outputSize, int priorKnowledgeSize, int timeStep,
                                Vector[][] dataset, Vector[] labels, int batchSize, int iteration) throws Exception {
        RNNTester tester = new RNNTester(inputSize, outputSize, priorKnowledgeSize, timeStep);
        DataGetter<Vector[]> xGetter = new DataGetter<>() {
            @Override
            public Vector[] at(int i) {
                return dataset[i];
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<>() {
            @Override
            public Vector at(int i) {
                return labels[i];
            }

            @Override
            public int size() {
                return dataset.length;
            }
        };

        // Loading !
        RNN rnn = new RNN(inputSize, priorKnowledgeSize, outputSize, timeStep, xGetter, yGetter);
        rnn.loadParameters(tester.getWhh(), tester.getWhx(), tester.getWyh(), tester.getBh(), tester.getBy());

        // Preparing test parameters!
        tester.train(dataset, labels, batchSize, iteration, 0.001);

        // read logs and test all parameters!
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./log.txt"));
        ObjectInputStream paramsOis = new ObjectInputStream(new FileInputStream("./params.txt"));

        boolean passed =
                rnn.train(0.001, iteration, batchSize,
                        false, (List<TestingObject>) ois.readObject(), (List<TestingObject>) paramsOis.readObject());
        ois.close();

        Assertions.assertTrue(passed);
    }
}