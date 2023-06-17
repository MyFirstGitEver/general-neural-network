package org.example.neural;

import org.example.Pair;
import org.example.Vector;

import java.io.Serializable;

public class Matrix implements Serializable {
    private final double[][] entries;

    public Matrix(String data) {
        String[] lines = data.split("\2");
        this.entries = new double[lines.length][];

        for(int j=0;j<lines.length;j++) {
            String line = lines[j];

            String[] numbers = line.split("\t");
            double[] numbersData = new double[numbers.length];

            for(int i=0;i<numbers.length;i++) {
                numbersData[i] = Double.parseDouble(numbers[i]);
            }

            entries[j] = numbersData;
        }
    }

    public Matrix(Vector v, boolean columnVector) {
        if(columnVector) {
            entries = new double[v.size()][1];

            for(int i=0;i<v.size();i++) {
                entries[i][0] = v.x(i);
            }
        }
        else {
            entries = new double[1][v.size()];

            for(int i=0;i<v.size();i++) {
                entries[0][i] = v.x(i);
            }
        }
    }

    public Matrix(double[]... data) {
        entries = data;
    }

    public Matrix(Vector[] twoD) {
        entries = new double[twoD.length][twoD[0].size()];

        for(int i=0;i<twoD.length;i++) {
            for(int j=0;j<twoD[0].size();j++) {
                entries[i][j] = twoD[i].x(j);
            }
        }
    }

    public Matrix(int width, int height) {
        entries = new double[width][height];
    }

    public void randomise(double scaleFactor) {
        int m = entries.length, n = entries[0].length;

        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                entries[i][j] = Math.random() * scaleFactor;
            }
        }
    }

    public Vector[] mul(Matrix mat) throws Exception {
        Pair<Integer, Integer> shape = mat.shape();

        if(entries[0].length != shape.first) {
            throw new Exception("Can't multiply these two matrices");
        }

        int common = shape.first;
        Vector[] newMat = new Vector[entries.length];

        for(int i=0;i<entries.length;i++) {
            newMat[i] = new Vector(shape.second);

            for(int j=0;j<shape.second;j++) {
                double total = 0;

                for(int k=0;k<common;k++) {
                    total += entries[i][k] * mat.entries[k][j];
                }

                newMat[i].setX(j, total);
            }
        }

        return newMat;
    }

    public Pair<Integer, Integer> shape() {
        return new Pair<>(entries.length, entries[0].length);
    }

    public boolean identical(Vector[] mat) {
        if(entries.length != mat.length || entries[0].length != mat[0].size()) {
            return false;
        }

        for(int i=0;i<mat.length;i++) {
            for(int j=0;j<mat[0].size();j++) {
                if(mat[i].x(j) != entries[i][j]) {
                    return false;
                }
            }
        }

        return true;
    }

    public void reset() {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] = 0;
            }
        }
    }

    public void add(Vector[] mat) throws Exception {
        if(entries.length != mat.length || entries[0].length != mat[0].size()) {
            throw new Exception("Can't add these two matrices");
        }
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] += mat[i].x(j);
            }
        }
    }

    public void add(Matrix matrix) throws Exception {
        Pair<Integer, Integer> shape = matrix.shape();

        if(entries.length != shape.first || entries[0].length != shape.second) {
            throw new Exception("Can't add these two matrices");
        }
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] += matrix.entries[i][j];
            }
        }
    }

    public Matrix square() {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] *= entries[i][j];
            }
        }

        return this;
    }

    public Matrix copy() {
        double[][] newMat = new double[entries.length][entries[0].length];

        for(int i=0;i<newMat.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                newMat[i][j] = entries[i][j];
            }
        }

        return new Matrix(newMat);
    }

    public Vector[] vectorize(boolean byRow) {
        if(byRow) {
            Vector[] vectors = new Vector[entries.length];

            for(int i=0;i<entries.length;i++) {
                vectors[i] = new Vector(entries[i]);
            }

            return vectors;
        }

        Vector[] vectors = new Vector[entries[0].length];

        for(int i=0;i<entries[0].length;i++) {
            vectors[i] = new Vector(entries.length);

            for(int j=0;j<entries.length;j++) {
                vectors[i].setX(j, entries[j][i]);
            }
        }

        return vectors;
    }

    public void normalizeByRow() {
        for(int i=0;i<entries.length;i++) {
            Vector v = new Vector(entries[i]);
            v.normalise();

            for (int j=0;j<v.size();j++) {
                entries[i][j] = v.x(j);
            }
        }
    }

    public Matrix scale(double scale) {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j]  *= scale;
            }
        }

        return this;
    }

    public void divideBy(double scale) {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] /= scale;
            }
        }
    }

    public Matrix hadamardDivideCopy(Matrix mat, double eps) throws Exception {
        int m = entries.length;
        int n = entries[0].length;

        if(!sameShape(mat)) {
            throw new Exception("Failed to divide!");
        }

        double[][] newMat = new double[m][n];

        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                newMat[i][j] = entries[i][j] / (mat.at(i, j) + eps);
            }
        }

        return new Matrix(newMat);
    }

    public void selfSubtract(Matrix mat) throws Exception {
        int m = entries.length;
        int n = entries[0].length;

        if(!sameShape(mat)) {
            throw new Exception("Failed to subtract!");
        }

        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                entries[i][j] -= mat.at(i, j);
            }
        }
    }

    public Matrix sqrtCopy() {
        int m = entries.length;
        int n = entries[0].length;

        double[][] newMat = new double[m][n];

        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                newMat[i][j] = Math.sqrt(entries[i][j]);
            }
        }

        return new Matrix(newMat);
    }

    public static Matrix transpose(Vector[] mat) {
        double[][] answer = new double[mat[0].size()][mat.length];

        for(int i=0;i<mat.length;i++){
            for(int j=0;j<mat[0].size();j++) {
                answer[j][i] = mat[i].x(j);
            }
        }

        return new Matrix(answer);
    }

    public double at(int i, int j) {
        return entries[i][j];
    }

    @Override
    public String toString() {
        int m = entries.length;
        int n = entries[0].length;

        StringBuilder builder = new StringBuilder();
        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                builder.append(entries[i][j]).append("\t");
            }

            builder.append("\2");
        }

        return builder.toString();
    }

    public Matrix concatToLeftCopy(Matrix mat) throws Exception {
        Pair<Integer, Integer> shape = mat.shape();

        int m = entries.length;
        int n = entries[0].length;

        if(shape.first != m) {
            throw new Exception("Can't concat since the two does not have same row dimensions");
        }

        int newN = n + shape.second;

        double[][] newMat = new double[m][newN];
        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                newMat[i][j] = entries[i][j];
            }

            for(int j=n;j<newN;j++) {
                newMat[i][j] = mat.entries[i][j - n];
            }
        }

        return new Matrix(newMat);
    }

    public boolean sameShape(Matrix mat) {
        int m = entries.length;
        int n = entries[0].length;

        Pair<Integer, Integer> shape = mat.shape();

        return m == shape.first && n == shape.second;
    }
}