package org.example.networks;

import org.example.Vector;
import org.example.chooser.Concater;
import org.example.chooser.OutputChooser;
import org.example.layers.CompleteEdgeLayer;
import org.example.loss.Loss;
import org.example.neural.Matrix;
import org.example.others.DataGetter;

import java.io.BufferedReader;

public class NeuronNetwork {
    private final CompleteEdgeLayer[] edgeLayers;
    private final Loss loss;

    private final Concater concater;

    private final OutputChooser outputChooser;

    private final DataGetter<Vector[]> xGetter;

    private final DataGetter<Vector> yGetter;

    public NeuronNetwork(
            CompleteEdgeLayer[] edgeLayers,
            Loss loss,
            Concater concater,
            OutputChooser outputChooser,
            DataGetter<Vector[]> xGetter,
            DataGetter<Vector> yGetter) throws Exception {

        this.edgeLayers = edgeLayers;
        this.loss = loss;
        this.concater = concater;
        this.outputChooser = outputChooser;

        this.xGetter = xGetter;
        this.yGetter = yGetter;

        for(int i=0;i<edgeLayers.length - 1;i++) {
            if(edgeLayers[i].getASize() != edgeLayers[i + 1].getXSize()) {
                throw new Exception("This network is invalid!");
            }
        }
    }

    public void forward(Vector[] inputs) throws Exception {
        concater.setInputs(inputs);

        for(int i = 0;i < edgeLayers.length;i++) {
            Vector input = concater.of(i);

            if(input != null) {
                edgeLayers[i].concatInput(input);
            }

            edgeLayers[i].forward();
        }
    }

    public Vector[] getOutputs() {
        Vector[] outputs = new Vector[outputChooser.count()];

        int index = 0;
        for(int i=0;i<edgeLayers.length;i++) {
            if(outputChooser.choose(i)) {
                outputs[index] = edgeLayers[i].getFinalOutput();
                index++;
            }
        }

        return outputs;
    }

    public boolean train(double learningRate, int iteration, int batchSize, BufferedReader reader) throws Exception {
        if(batchSize > xGetter.size()) {
            throw new Exception("Failed to train because dataset is smaller than batch");
        }

        for(int iter=0;iter<iteration;iter++) {
            for(int i = 0;i < xGetter.size();i += batchSize) {
                int from = i;
                int to = i + Math.min(batchSize, xGetter.size() - i) - 1;

                for(int j = from; j <= to; j++) {
                    backward(xGetter.at(j), yGetter.at(j));
                }

                clip();

                if(reader != null) {
                    if(!valid(reader.readLine())) {
                        return false;
                    }
                }

                for(CompleteEdgeLayer edgeLayer : edgeLayers) {
                    edgeLayer.update(learningRate, outputChooser.count(), xGetter.size());
                }
            }
        }

        return true;
    }

    public void clip() {
        for (CompleteEdgeLayer edgeLayer : edgeLayers) {
            edgeLayer.clip();
        }
    }

    public void loadTestingToLayer(Vector[] W, Vector B, int layerId) {
        edgeLayers[layerId].load(W, B);
    }

    private boolean valid(String line) {
        String[] bAndW = line.split("\3");

        Vector[] b = bReader(bAndW[0]);
        Matrix[] W = wReader(bAndW[1]);

        int index = 0;
        for(int i=edgeLayers.length-1;i>=0;i--) {
            if(!edgeLayers[i].valid(W[index], b[index])) {
                return false;
            }

            index++;
        }

        return true;
    }

    private Vector[] bReader(String bStr) {
        String[] vecs = bStr.split("\2");
        Vector[] b = new Vector[vecs.length];

        for(int i=0;i<vecs.length;i++) {
            b[i] = new Vector(vecs[i]);
        }

        return b;
    }

    private Matrix[] wReader(String wStr) {
        String[] mats = wStr.split("\4");
        Matrix[] w = new Matrix[mats.length];

        for(int i=0;i<mats.length;i++) {
            w[i] = new Matrix(mats[i]);
        }

        return w;
    }

    private void backward(Vector[] inputs, Vector label) throws Exception {
        forward(inputs);
        Vector[] outputs = getOutputs();

        Vector firstEigenDeltas = new Vector(outputs[0].size());
        for(Vector output : outputs) {
            firstEigenDeltas.add(loss.dA(output, label));
        }

        edgeLayers[edgeLayers.length - 1].setEigenDeltaForLast(firstEigenDeltas);

        for(int i = edgeLayers.length - 1; i >= 0;i--) {
            edgeLayers[i].backward();
        }
    }
}