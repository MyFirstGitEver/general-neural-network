package org.example.chooser.outputchoosers;

import org.example.Vector;

public record RNNOutputProcessor(int count, int outputSize) implements OutputProcessor {

    @Override
    public boolean choose(int i) {
        return (i % 2 == 1);
    }

    @Override
    public Vector preprocess(Vector output) {
        return output.subVec(0, outputSize);
    }
}