package org.example.layers.weighted;

import org.example.Pair;
import org.example.Vector;
import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.layers.Layer;
import org.example.networks.Neuron;
import org.example.neural.Matrix;
import org.example.others.Learnable;

import java.util.HashMap;

public abstract class WeightedLayer extends Layer {
    private final long hashFactor;
    protected final Learnable[] bParams;
    protected final HashMap<Long, Learnable> wParamsOf;

    public WeightedLayer(int inputSize, int outputSize, Layer previousLayer, int... args) {
        super(inputSize, outputSize, previousLayer, args);

        hashFactor = outputSize;
        bParams = new Learnable[this.Y.length];
        wParamsOf = new HashMap<>();

        buildBiases();
        buildWeights();
    }

    public WeightedLayer(WeightedLayer weightedLayer, Layer previousLayer) {
        super(weightedLayer.X.length, weightedLayer.Y.length, previousLayer);

        hashFactor = weightedLayer.hashFactor;
        bParams = weightedLayer.bParams;
        wParamsOf = weightedLayer.wParamsOf;
    }

    abstract public WeightedLayer copy(Layer lastLayer);

    protected void buildBiases() {
        for(int i = 0;i < bParams.length; i++) {
            bParams[i] = new Learnable(Math.random());
        }
    }

    protected void buildWeights() {
        for(int destination=0;destination<Y.length;destination++) {
            NumberGenerator generator = Y[destination].getBackwardNeurons();
            Integer source;

            while((source = generator.next()) != null) {
                wParamsOf.put(hash(source, destination), new Learnable(Math.random()));
            }
        }
    }

    @Override
    public void forward() {
        for(int j = 0; j< Y.length; j++) {
            Neuron y = Y[j];

            double weighted = 0;
            Integer source;
            NumberGenerator generator = y.getBackwardNeurons();

            while((source = generator.next()) != null) {
                weighted += X[source].getValue() * wParamsOf.get(hash(source, j)).getValue();
            }

            y.setValue(weighted + bParams[j].getValue());
        }
    }

    @Override
    public void backward() {
        for(int j = 0; j< Y.length; j++) {
            bParams[j].setDValue(bParams[j].getDValue() + eigenDelta[j]);
        }

        for(int i=0;i<X.length;i++) {
            Neuron x = X[i];

            double total = 0;
            Integer destination;
            NumberGenerator generator = x.getForwardNeurons();

            while((destination = generator.next()) != null) {
                Learnable w = wParamsOf.get(hash(i, destination));
                w.setDValue(w.getDValue() + eigenDelta[destination] * x.getValue());

                total += eigenDelta[destination] * wParamsOf.get(hash(i, destination)).getValue();
            }

            if(previousLayer != null) {
                previousLayer.getEigenDelta()[i] = total;
            }
        }
    }

    public void update(double learningRate, int outputCount, int datasetSize) {
        for(Learnable b : bParams) {
            b.selfUpdate(learningRate, outputCount, datasetSize);
        }

        for(int i=0;i<X.length;i++) {
            Integer destination;
            NumberGenerator generator = X[i].getForwardNeurons();

            while((destination = generator.next()) != null) {
                wParamsOf.get(hash(i, destination)).selfUpdate(learningRate, outputCount, datasetSize);
            }
        }
    }

    protected long hash(int source, int destination) {
        return source * hashFactor + destination;
    }

    public void load(Vector[] w, Vector b, int offset) {
        int outputSize = w.length;
        int inputSize = w[0].size();

        for(int i=0;i<inputSize;i++) {
            for(int j=0;j<outputSize;j++) {
                if(wParamsOf.get(hash(i + offset, j)) == null) {
                    int m = 3;
                }

                wParamsOf.get(hash(i + offset, j)).setValue(w[j].x(i));
            }
        }

        if(b == null) {
            return; // stop here!
        }

        for(int i=0;i<outputSize;i++) {
            bParams[i].setValue(b.x(i));
        }
    }

    public boolean valid(Matrix dW, Vector dB, int offset) {
        Pair<Integer, Integer> shape = dW.shape();

        int outputSize = shape.first;
        int inputSize = shape.second;

        for(int i=0;i<inputSize;i++) {
            for(int j=0;j<outputSize;j++) {
                if(Math.abs(wParamsOf.get(hash(i + offset, j)).getDValue() - dW.at(j, i)) > 4e-3) {
                    return false;
                }
            }
        }

        if(dB == null) {
            return true; // stop here
        }

        for(int i=0;i<outputSize;i++) {
            if(Math.abs(bParams[i].getDValue() - dB.x(i)) > 2e-3) {
                return false;
            }
        }

        return true;
    }

    public void clip() {
        clipDB();
        clipDW();
    }

    private void clipDW() {
        for(int i=0;i<Y.length;i++) {
            Neuron outputNeu = Y[i];
            NumberGenerator generator = outputNeu.getBackwardNeurons();

            Vector dW = new Vector(generator.count());

            int index = 0;
            Integer source;
            while((source = generator.next()) != null) {
                dW.setX(index, wParamsOf.get(hash(source, i)).getDValue());
                index++;
            }

            dW.normalise();

            index = 0;
            while((source = generator.next()) != null) {
                wParamsOf.get(hash(source, i)).setDValue(dW.x(index));
                index++;
            }
        }
    }

    private void clipDB() {
        Vector db = new Vector(bParams.length);

        for(int i=0;i<bParams.length;i++) {
            db.setX(i, bParams[i].getDValue());
        }

        db.normalise();

        for(int i=0;i<bParams.length;i++) {
            bParams[i].setDValue(db.x(i));
        }
    }
}