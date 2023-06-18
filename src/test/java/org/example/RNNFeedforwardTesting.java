package org.example;

import org.example.networks.RNN;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

// Test 1 : 87 / 313
public class RNNFeedforwardTesting {
    public static Stream<Arguments> cases() {
        return Stream.of(oneHundredArgs());
    }

    private static Arguments[] oneHundredArgs() {
        Arguments[] args = new Arguments[30];

        for(int i=0;i<args.length;i++) {
            Vector[] data = vectors((int) (Math.random() * 300 + 15), (int) (Math.random() * 300 + 15));

            args[i] = Arguments.of(
                    data[0].size(),
                    2,
                    (int) (Math.random() * 300 + 15),
                    data.length, data);
        }

        return args;
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

    @ParameterizedTest
    @MethodSource("cases")
    public void feedforwardTesting(int inputSize, int outputSize, int priorKnowledgeSize, int timeStep, Vector[] input)
            throws Exception {
        RNNTester tester = new RNNTester(inputSize, outputSize, priorKnowledgeSize, timeStep);
        Vector[] outputs = tester.feedforward(input);

        RNN rnn = new RNN(inputSize, priorKnowledgeSize, outputSize, timeStep, null,null);
        rnn.loadParameters(tester.getWhh(), tester.getWhx(), tester.getWyh(), tester.getBh(), tester.getBy());
        Vector[] rnnOutputs = rnn.outputs(input);

        for(int i=0;i<outputs.length;i++) {
            Vector result = rnnOutputs[i].subVec(0, outputSize);
            Assertions.assertTrue(outputs[i].identical(result, 0.005));
        }
    }
}