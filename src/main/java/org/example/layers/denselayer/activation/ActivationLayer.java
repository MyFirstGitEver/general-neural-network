package org.example.layers.denselayer.activation;

import org.example.chooser.NumberGenerator;
import org.example.layers.Layer;
import org.example.activators.Activator;
import org.example.networks.Neuron;
public abstract class ActivationLayer extends Layer {
    private final Activator activator;

    public ActivationLayer(int inputSize, int outputSize, Layer previousLayer, Activator activator) {
        super(inputSize, outputSize, previousLayer);

        this.activator = activator;
    }

    @Override
    public void forward() {
        for(int j = 0; j < Y.length; j++) {
            Neuron y = Y[j];
            NumberGenerator backwardId = y.getBackwardNeurons();

            y.setValue(activator.g(X, backwardId, j));
        }
    }

    @Override
    public void backward() {
        for(int i = 0; i < X.length; i++) {
            double total = 0;

            NumberGenerator generator = X[i].getForwardNeurons();
            Integer destination;

            while((destination = generator.next()) != null) {
                NumberGenerator backwardIds = Y[destination].getBackwardNeurons();

                total += eigenDelta[destination] * activator.dg(X, backwardIds, i, destination);
            }

            previousLayer.getEigenDelta()[i] = total;
        }
    }
}