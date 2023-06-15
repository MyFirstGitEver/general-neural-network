package org.example.chooser;

public abstract class NumberGenerator {
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

    public abstract Integer next();

    public abstract Integer back();
    public abstract int count();
}