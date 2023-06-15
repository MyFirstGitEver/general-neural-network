package org.example.others;

public class Learnable {
    private final static double beta = 0.9f;
    private final static double beta2 = 0.999f;

    public Learnable() {
    }

    public Learnable(double value) {
        this.value = value;
    }

    public void selfUpdate(double learningRate, int outputCount, int datasetSize) {
        dValue /= (outputCount * datasetSize);

        double firstMoment = first * beta + dValue * (1 - beta);
        double secondMoment = second * beta2 + dValue * dValue * (1 - beta2);

        value -= firstMoment / (Math.sqrt(secondMoment) + 10e-8f) * learningRate;

        first = firstMoment;
        second = secondMoment;

        dValue = 0;
    }

    private double value, dValue, first, second;

    public double getValue() {
        return value;
    }

    public double getDValue() {
        return dValue;
    }

    public double getFirst() {
        return first;
    }

    public double getSecond() {
        return second;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public void setDValue(double dValue) {
        this.dValue = dValue;
    }

    public void setFirst(double first) {
        this.first = first;
    }

    public void setSecond(double second) {
        this.second = second;
    }
}