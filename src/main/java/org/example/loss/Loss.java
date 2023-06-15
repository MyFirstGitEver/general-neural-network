package org.example.loss;

import org.example.Vector;

public interface Loss {
    double loss(Vector a, Vector y) throws Exception;
    Vector dA(Vector a, Vector y);
}
