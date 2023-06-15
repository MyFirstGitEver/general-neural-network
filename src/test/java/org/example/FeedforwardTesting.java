package org.example;

import org.example.activators.FeedForwardLinearActivator;
import org.example.activators.FeedForwardSoftmaxActivator;
import org.example.activators.FeedforwardReluActivator;
import org.example.loss.MSE;
import org.example.networks.FeedforwardNetwork;
import org.example.neural.*;
import org.example.activators.Activator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

public class FeedforwardTesting {
    public static Stream<Arguments> cases() {
        Vector[] x = oneHundredVectors(null);
        Vector[] v = oneHundredVectors(x);

        Arguments[] set = new Arguments[100];

        for(int i=0;i<100;i++) {
            set[i] = Arguments.of(x[i], v[i], (int) (3 + Math.random() * 15));
        }

        return Stream.of(set);
    }

    public static Vector[] oneHundredVectors(Vector[] sample) {
        Vector[] set = new Vector[100];

        for(int j=0;j<100;j++) {
            int size;
            if(sample != null) {
                size = sample[j].size();
            }
            else {
                size = (int) (3 + Math.random() * 15);
            }

            Vector v = new Vector(size);
            set[j] = v;

            for(int i=0;i<v.size();i++) {
                v.setX(i, -15.0 + Math.random() * 30.0);
            }
        }

        return set;
    }

    @ParameterizedTest
    @MethodSource("cases")
    public void forwardTesting(Vector x, Vector v,int outputSize) throws Exception {
        int featureSize = x.size();

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(new DenseLayer[]{
                new DenseLayer(new ReluActivation(featureSize, 10)),
                new DenseLayer(new ReluActivation(10, 5)),
                new DenseLayer(new SoftMaxActivation(5, outputSize))
        }, new org.example.neural.MSE(), null, null);

        FeedforwardNetwork network = new FeedforwardNetwork(new int[] {
                featureSize, 10, 5, outputSize
        }, new Activator[] {
                new FeedforwardReluActivator(),
                new FeedforwardReluActivator(),
                new FeedForwardSoftmaxActivator()
        }, new MSE(), null, null);

        network.loadTestingToLayer(model.w(0), model.b(0), 0);
        network.loadTestingToLayer(model.w(1), model.b(1), 1);
        network.loadTestingToLayer(model.w(2), model.b(2), 2);

        Vector out = model.predict(v);
        Vector out2 = network.output(v);

        Assertions.assertEquals(out.size(), out2.size());

        for(int i=0;i<out.size();i++) {
            Assertions.assertTrue(Math.abs(out.x(i) - out2.x(i)) < 0.00001);
        }
    }
}