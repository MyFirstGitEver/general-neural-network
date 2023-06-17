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
        Arguments[] args = new Arguments[300];

        for(int i=0;i<args.length;i++) {
            Vector[] data = vectors(randomTimeStep(), (int) (Math.random() * 2 + 3));

            Vector label = randomOneHotVector(2);

            args[i] = Arguments.of(
                    data[0].size(),
                    2,
                    (int) (Math.random() * 2 + 13),
                    data.length, data, label);
        }


        return args;
    }

    private static int randomTimeStep() {
        if(Math.random() > 0.9) {
            return (int) (Math.random() * 3600 + 400);
        }
        else if(Math.random() > 0.7) {
            return (int) (Math.random() * 100 + 60);
        }

        return (int) (Math.random() * 5 + 1);
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
                                Vector[] input, Vector label) throws Exception {
        RNNTester tester = new RNNTester(inputSize, outputSize, priorKnowledgeSize, timeStep);
        tester.backward(input, label);

        DataGetter<Vector[]> xGetter = new DataGetter<>() {
            @Override
            public Vector[] at(int i) {
                return input;
            }

            @Override
            public int size() {
                return 1;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<>() {
            @Override
            public Vector at(int i) {
                return label;
            }

            @Override
            public int size() {
                return 1;
            }
        };

        RNN rnn = new RNN(inputSize, priorKnowledgeSize, outputSize, timeStep, xGetter, yGetter);
        rnn.loadParameters(tester.getWhh(), tester.getWhx(), tester.getWyh(), tester.getBh(), tester.getBy());
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./log.txt"));
        boolean passed =
                rnn.train(0.001, 1, 1, false, (List<TestingObject>) ois.readObject());
        ois.close();

        Assertions.assertTrue(passed);
    }
}