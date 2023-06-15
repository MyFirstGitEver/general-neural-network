package org.example.others;

public interface DataGetter<X> {
    X at(int i);
    int size();
}