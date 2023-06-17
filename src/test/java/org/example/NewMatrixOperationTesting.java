package org.example;

import org.example.neural.Matrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

public class NewMatrixOperationTesting {
    public static Stream<Arguments> sqrtCopy() {
        return Stream.of(
                Arguments.of(new double[][]
                                {
                                        {4, 9, 1, 4},
                                        {1, 0, 4, 16},
                                        {9, 4, 1, 0}
                                },
                        new double[][] {
                                {2, 3, 1, 2},
                                {1, 0, 2, 4},
                                {3, 2, 1, 0}
                        }),
                Arguments.of(new double[][]
                                {
                                        {2, 4, 0, 4, 16},
                                        {9, 0, 1, 1, 5},
                                        {9, 4, 4, 0, 3}
                                },
                        new double[][] {
                                {Math.sqrt(2), 2, 0, 2, 4},
                                {3, 0, 1, 1, Math.sqrt(5)},
                                {3, 2, 2, 0, Math.sqrt(3)}
                        }
        ));
    }

    public static Stream<Arguments> selfSubtract() {
        return Stream.of(
                Arguments.of(
                        new double[][] {
                                {3, 9, 5, 15, 4},
                                {22, 6, 5, 11, 4}
                        },
                        new double[][] {
                                {4, 6, 15, 22, 8},
                                {0, -3, -9, -15, 4}
                        },
                        new double[][] {
                                {-1, 3, -10, -7, -4},
                                {22, 9, 14, 26, 0}
                        }));
    }


    public static Stream<Arguments> hadamardDivideCopy() {
        return Stream.of(
                Arguments.of(
                        new double[][] {
                                {3, 9, 5, 15, 4},
                                {22, 6, 5, 11, 4}
                        },
                        new double[][] {
                                {4, 6, 15, 22, 8},
                                {1, -3, -9, -15, 4}
                        },
                        new double[][] {
                                {0.75, 1.5, 1.0 / 3.0, 15.0 / 22.0, 0.5},
                                {22, -2, -5.0 / 9, -11.0 / 15, 1}
                        }));
    }

    public static Stream<Arguments> concatToLeft() {
        return Stream.of(
                Arguments.of(
                        new double[][] {
                                {3, 1, -15, 4},
                                {0, 8, 7, -3}
                        },
                        new double[][] {
                                {9, 11, 13},
                                {17, 25, -3}
                        },
                        new double[][] {
                                {3, 1, -15, 4, 9, 11, 13},
                                {0, 8, 7, -3, 17, 25, -3}
                        }),
                Arguments.of(
                        new double[][] {
                                {3, 1, -15, 4},
                                {0, 8, 7, 15},
                                {0, 8, 0, 15},
                        },
                        new double[][] {
                                {9, 11, 13},
                                {0, 25, -3},
                                {17, 25, -3},
                        },
                        new double[][] {
                                {3, 1, -15, 4, 9, 11, 13},
                                {0, 8, 7, 15, 0, 25, -3},
                                {0, 8, 0, 15, 17, 25, -3},
                        })
        );
    }

    @ParameterizedTest
    @MethodSource("selfSubtract")
    public void selfSubtractTesting(double[][] mat, double[][] subtractor, double[][] expected) throws Exception {
        Matrix matrix = new Matrix(mat);

        matrix.selfSubtract(new Matrix(subtractor));

        Assertions.assertTrue(matrix.sameShape(new Matrix(expected)));

        for(int i=0;i<expected.length;i++) {
            for(int j=0;j<expected[0].length;j++) {
                Assertions.assertEquals(matrix.at(i, j), expected[i][j]);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("sqrtCopy")
    public void sqrtCopyTesting(double[][] mat, double[][] expected) {
        Matrix matrix = new Matrix(mat);

        Matrix result = matrix.sqrtCopy();

        Assertions.assertTrue(result.sameShape(new Matrix(expected)));

        for(int i=0;i<expected.length;i++) {
            for(int j=0;j<expected[0].length;j++) {
                Assertions.assertEquals(result.at(i, j), expected[i][j]);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("hadamardDivideCopy")
    public void hadamardTesting(double[][] mat, double[][] divisor, double[][] expected) throws Exception {
        Matrix matrix = new Matrix(mat);

        Matrix result = matrix.hadamardDivideCopy(new Matrix(divisor), 0.000000001);

        Assertions.assertTrue(result.sameShape(new Matrix(expected)));

        for(int i=0;i<expected.length;i++) {
            for(int j=0;j<expected[0].length;j++) {
                Assertions.assertTrue(Math.abs(expected[i][j] - result.at(i, j)) < 0.00001);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("concatToLeft")
    public void concatToLeftTesting(double[][] first, double[][] second, double[][] answer) throws Exception {
        Matrix firstMat = new Matrix(first);
        Matrix secondMat = new Matrix(second);

        Matrix answerMat = new Matrix(answer);
        Matrix actualMat = firstMat.concatToLeftCopy(secondMat);

        Assertions.assertTrue(answerMat.sameShape(actualMat));

        Pair<Integer, Integer> shape = answerMat.shape();

        for(int i=0;i<shape.first;i++) {
            for(int j=0;j<shape.second;j++) {
                Assertions.assertEquals(answerMat.at(i, j), actualMat.at(i, j));
            }
        }
    }
}
