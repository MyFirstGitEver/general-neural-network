package org.example.networks;

import org.example.TestingObject;
import org.example.Vector;
import org.example.chooser.concaters.Concater;
import org.example.chooser.numbergenerators.NumberGenerator;
import org.example.chooser.outputchoosers.OutputProcessor;
import org.example.layers.CompleteForwarder;
import org.example.loss.Loss;
import org.example.neural.Matrix;
import org.example.others.DataGetter;

import java.util.List;

public class NeuronNetwork {
    private final CompleteForwarder[] forwarders;
    private final Loss loss;

    private final Concater concater;

    private final OutputProcessor outputProcessor;

    private final DataGetter<Vector[]> xGetter;

    private final DataGetter<Vector> yGetter;

    private final NumberGenerator updateParamsInstructor;

    public NeuronNetwork(
            CompleteForwarder[] forwarders,
            Loss loss,
            Concater concater,
            OutputProcessor outputProcessor,
            DataGetter<Vector[]> xGetter,
            DataGetter<Vector> yGetter,
            NumberGenerator updateParamsInstructor) throws Exception {

        this.forwarders = forwarders;
        this.loss = loss;
        this.concater = concater;
        this.outputProcessor = outputProcessor;

        this.updateParamsInstructor = updateParamsInstructor;

        this.xGetter = xGetter;
        this.yGetter = yGetter;

        for(int i = 0; i< forwarders.length - 1; i++) {
            if(forwarders[i].getASize() != forwarders[i + 1].getXSize()) {
                throw new Exception("This network is invalid!");
            }
        }
    }

    public void forward(Vector[] inputs) throws Exception {
        concater.setInputs(inputs);

        for(int i = 0; i < forwarders.length; i++) {
            Vector input = concater.of(i);

            if(input != null) {
                forwarders[i].concatInput(input);
            }

            forwarders[i].forward();
        }
    }

    public Vector[] getOutputs(int lastK) {
        Vector[] outputs = new Vector[lastK];

        int index = outputs.length - 1;
        for(int i = forwarders.length - 1; i>=0 && index >= 0; i--) {
            if(outputProcessor.choose(i)) {
                outputs[index] = forwarders[i].getFinalOutput();
                index--;
            }
        }

        return outputs;
    }

    public boolean train(double learningRate,
                         int iteration,
                         int batchSize,
                         int maxToPrint,
                         List<TestingObject> deltaTests,
                         List<TestingObject> updatedParameters,
                         boolean printCost) throws Exception {
        if(batchSize > xGetter.size()) {
            throw new Exception("Failed to train because dataset is smaller than batch");
        }

        int gradientIndex = 0;
        int paramIndex = 0;

        for(int iter=0;iter<iteration;iter++) {
            if(iter % maxToPrint == 0 && printCost) {
                System.out.println(iter + " iterations have passed. Cost: "  + cost());
            }

            for(int i = 0;i < xGetter.size();i += batchSize) {
                int from = i;
                int to = i + Math.min(batchSize, xGetter.size() - i) - 1;

                for(int j = from; j <= to; j++) {
                    backward(xGetter.at(j), yGetter.at(j));
                }

                TestingObject delta = null;
                TestingObject params = null;

                if(deltaTests != null) {
                    delta = deltaTests.get(gradientIndex);
                    gradientIndex++;
                }

                if(updatedParameters != null) {
                    params = updatedParameters.get(paramIndex);
                    paramIndex++;
                }

                //clip();

                if(delta != null) {
                    int index = 0;

                    Matrix[] W = delta.getdW();
                    Vector[] B = delta.getdB();
                    int[] offsets = delta.getOffset();

                    NumberGenerator layerChooser = delta.getLayerChooser();
                    Integer test;

                    while((test = layerChooser.next()) != null) {
                        if(!forwarders[test].valid(W[index], B[index], offsets[index])) {
                            return false;
                        }

                        index++;
                    }
                }

                Integer next;

                while((next = updateParamsInstructor.next()) != null) {
                    forwarders[next].update(learningRate, outputProcessor.count(), to - from + 1);
                }

                if(params != null) {
                    int index = 0;

                    Matrix[] W = params.getdW();
                    Vector[] B = params.getdB();
                    int[] offsets = params.getOffset();

                    NumberGenerator layerChooser = params.getLayerChooser();
                    Integer test;

                    while((test = layerChooser.next()) != null) {
                        if(!forwarders[test].validParams(W[index], B[index], offsets[index])) {
                            return false;
                        }

                        index++;
                    }
                }
            }
        }

        return true;
    }

    public void clip() {
        for (CompleteForwarder edgeLayer : forwarders) {
            edgeLayer.clip();
        }
    }

    public void loadTestingToLayer(Vector[] W, Vector B, int layerId, int offset) {
        forwarders[layerId].load(W, B, offset);
    }

    public int outputCount() {
        return outputProcessor.count();
    }

    private void backward(Vector[] inputs, Vector label) throws Exception {
        forward(inputs);

        for(int source = forwarders.length - 1; source>=0; source--) {
            if(!outputProcessor.choose(source)) {
                continue;
            }

            Vector output = forwarders[source].getFinalOutput();
            Vector firstEigenDeltas = loss.dA(outputProcessor.preprocess(output), label);

            forwarders[source].setEigenDeltas(firstEigenDeltas);

            for(int j = source; j >= 0;j--) {
                forwarders[j].backward();
            }
        }
    }

    public double cost() throws Exception {
        double total = 0;

        for (int i = 0; i < xGetter.size(); i++) {
            Vector[] x = xGetter.at(i);

            forward(x);

            for(int source = forwarders.length - 1; source>=0; source--) {
                if(!outputProcessor.choose(source)) {
                    continue;
                }

                Vector output = forwarders[source].getFinalOutput();
                total += loss.loss(output, yGetter.at(i));
            }
        }

        return total / xGetter.size() / outputProcessor.count();
    }
}