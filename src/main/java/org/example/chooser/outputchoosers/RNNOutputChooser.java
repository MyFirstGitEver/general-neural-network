package org.example.chooser.outputchoosers;

public class RNNOutputChooser implements OutputChooser {
    private final int count;

    public RNNOutputChooser(int count) {
        this.count = count;
    }

    @Override
    public boolean choose(int i) {
        return (i % 2 == 1);
    }

    @Override
    public int count() {
        return count;
    }
}
