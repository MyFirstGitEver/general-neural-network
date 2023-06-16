package org.example.chooser.concaters;

import org.example.Vector;

public class RNNConcater extends Concater {
    @Override
    public Vector of(int index) {
        if(index % 2 == 0) {
            return inputs[index / 2];
        }

        return null;
    }
}
