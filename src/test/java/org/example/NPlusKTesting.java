package org.example;

import org.example.chooser.numbergenerators.NPlusKGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

public class NPlusKTesting {
    public static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of(5, 0, new int[] {0, 1, 2, 3, 4}),
                Arguments.of(4, 2, new int[] {0, 1, 2, 3, 5}),
                Arguments.of(8, 15, new int[] {0, 1, 2, 3, 4, 5, 6, 7, 22}),
                Arguments.of(4, 15, new int[] {0, 1, 2, 3, 18})
        );
    }

    @ParameterizedTest
    @MethodSource("cases")
    public void NPlusK(int n, int k, int[] expected) {
        NPlusKGenerator generator = new NPlusKGenerator(0, n, n + k);
        Assertions.assertEquals(generator.count(), expected.length);

        Integer num;

        int index = 0;

        while((num = generator.next()) != null) {
            Assertions.assertEquals(num, expected[index]);
            index++;
        }
    }
}
