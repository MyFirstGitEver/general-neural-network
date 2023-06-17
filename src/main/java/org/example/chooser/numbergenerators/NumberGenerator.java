package org.example.chooser.numbergenerators;

import java.io.Serializable;

public abstract class NumberGenerator implements Serializable {
    protected final int targetValue;
    protected final int from;
    protected int currentValue;

    public NumberGenerator(int from, int targetValue) {
        this.targetValue = targetValue;
        this.from = from;

        currentValue = from;
    }

    public int getCurrentValue() {
        return currentValue;
    }

    public void reverse() {
        currentValue = targetValue - 1;
    }

    public void reset() {
        currentValue = from;
    }

    public abstract Integer next();

    public abstract Integer back();
    public abstract int count();
}