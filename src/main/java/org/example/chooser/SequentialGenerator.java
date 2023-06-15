package org.example.chooser;

public class SequentialGenerator extends NumberGenerator {

    public SequentialGenerator(int from, int targetValue) {
        super(from, targetValue);
    }

    @Override
    public Integer next() {
        int value = currentValue;

        currentValue++;
        if(currentValue > targetValue) {
            currentValue = from;
            return null;
        }

        return value;
    }

    @Override
    public Integer back() {
        int value = currentValue;

        currentValue--;
        if(currentValue < from - 1) {
            currentValue = from;
            return null;
        }

        return value;
    }

    @Override
    public int count() {
        return targetValue - from;
    }
}