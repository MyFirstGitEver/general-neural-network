package org.example;

import org.example.networks.RNN;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

public class RNNTesting {
    public static Stream<Arguments> cases() {
        return Stream.of(oneHundredArgs());
    }

    private static Arguments[] oneHundredArgs() {
        Arguments[] args = new Arguments[300];

        for(int i=0;i<args.length;i++) {
            Vector[] data = vectors((int) (Math.random() * 10 + 15), (int) (Math.random() * 100 + 3));

            args[i] = Arguments.of(
                    data[0].size(),
                    (int) (Math.random() * 10 + 15),
                    (int) (Math.random() * 10 + 15),
                    data.length, data);
        }

        return args;
    }

    public static Vector[] vectors(int num, Integer size) {
        Vector[] set = new Vector[num];

        for(int j=0;j<num;j++) {
            if(size == null) {
                size = (int) (Math.random() * 15 + 3);
            }

            Vector v = new Vector(size);
            set[j] = v;

            for(int i=0;i<v.size();i++) {
                v.setX(i, Math.random() * 0.001);
            }
        }

        return set;
    }

    @ParameterizedTest
    @MethodSource("cases")
    public void feedforwardTesting(int inputSize, int outputSize, int priorKnowledgeSize, int layers, Vector[] input)
            throws Exception {
        RNNTester tester = new RNNTester(inputSize, outputSize, priorKnowledgeSize, layers);
        Vector[] outputs = tester.feedforward(input);

        RNN rnn = new RNN(inputSize, priorKnowledgeSize, outputSize, layers, null,null);
        rnn.loadParameters(tester.getWhh(), tester.getWhx(), tester.getWyh(), tester.getBh(), tester.getBy());
        Vector[] rnnOutputs = rnn.outputs(input);

        for(int i=0;i<outputs.length;i++) {
            Assertions.assertTrue(outputs[i].identical(rnnOutputs[i], 0.001));
        }
    }
}