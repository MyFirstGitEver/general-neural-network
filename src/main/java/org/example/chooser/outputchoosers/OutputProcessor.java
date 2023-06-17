package org.example.chooser.outputchoosers;

import org.example.Vector;

public interface OutputProcessor {
    boolean choose(int i);

    Vector preprocess(Vector output);

    int count();
}