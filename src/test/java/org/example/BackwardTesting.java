package org.example;

import org.example.chooser.numbergenerators.BackwardGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

public class BackwardTesting {
    public static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of(6, 2, new int[] {6, 5, 4, 3}, new int[] {3, 4, 5, 6}),
                Arguments.of(4, 2, new int[] {4, 3}, new int[] {3, 4}),
                Arguments.of(5, 1, new int[] {5, 4, 3, 2}, new int[] {2, 3, 4, 5}));
    }

    @ParameterizedTest
    @MethodSource("cases")
    public void backwardTesting(int from, int targetValue, int[] forward, int[] backward) {
        BackwardGenerator generator = new BackwardGenerator(from, targetValue);

        Integer next;

        int index = 0;
        while((next = generator.next()) != null) {
            Assertions.assertEquals(next, forward[index]);
            index++;
        }

        Integer back;

        index = 0;
        generator.reverse();
        while((back = generator.back()) != null) {
            Assertions.assertEquals(back, backward[index]);
            index++;
        }
    }
}