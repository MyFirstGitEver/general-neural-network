package org.example.chooser.numbergenerators;

public class NPlusKGenerator extends NumberGenerator {
    private final int N;

    public NPlusKGenerator(int from, int N, int targetValue) {
        super(from, targetValue);

        this.N = N;
    }

    @Override
    public Integer next() {
        if(currentValue < N) {
            int value = currentValue;
            currentValue++;
            return value;
        }

        if(currentValue >= targetValue - 1) {
            currentValue = from;
            return null;
        }

        currentValue = targetValue - 1;
        return targetValue - 1;
    }

    @Override
    public Integer back() {
        if(currentValue == targetValue - 1) {
            currentValue = N - 1;
            return targetValue - 1;
        }

        if(currentValue == -1) {
            currentValue = from;
            return null;
        }

        int value = currentValue;
        currentValue--;
        return value;
    }

    @Override
    public int count() {
        return N + ((targetValue - N == 0) ? 0 : 1);
    }
}
