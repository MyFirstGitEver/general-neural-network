package org.example.chooser.numbergenerators;

public class BackwardGenerator extends NumberGenerator {
    public BackwardGenerator(int from, int targetValue) { // exclude target value
        super(from, targetValue);
    }

    @Override
    public Integer next() {
        int value = currentValue;

        currentValue--;
        if(currentValue == targetValue - 1) {
            currentValue = from;
            return null;
        }

        return value;
    }

    @Override
    public Integer back() {
        int value = currentValue;

        currentValue++;
        if(currentValue == from + 2) {
            currentValue = from;
            return null;
        }

        return value;
    }

    @Override
    public void reverse() {
        currentValue = targetValue + 1;
    }

    @Override
    public int count() {
        return from - targetValue + 1;
    }
}