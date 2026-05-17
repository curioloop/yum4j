package com.curioloop.yum4j.ssm.kalman.mle;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

public final class FixedParameters {

    private static final FixedParameters NONE = new FixedParameters(new int[0], new double[0]);

    private final int[] indices;
    private final double[] values;

    private FixedParameters(int[] indices, double[] values) {
        this.indices = indices;
        this.values = values;
    }

    public static FixedParameters none() {
        return NONE;
    }

    public static FixedParameters of(int[] indices, double[] values) {
        Objects.requireNonNull(indices, "indices");
        Objects.requireNonNull(values, "values");
        if (indices.length != values.length) {
            throw new IllegalArgumentException("indices and values must have the same length");
        }
        if (indices.length == 0) {
            return NONE;
        }
        Map<Integer, Double> ordered = new LinkedHashMap<>();
        for (int i = 0; i < indices.length; i++) {
            int index = indices[i];
            if (index < 0) {
                throw new IllegalArgumentException("fixed parameter index must be non-negative");
            }
            if (!Double.isFinite(values[i])) {
                throw new IllegalArgumentException("fixed parameter values must be finite");
            }
            if (ordered.put(index, values[i]) != null) {
                throw new IllegalArgumentException("duplicate fixed parameter index: " + index);
            }
        }
        int[] sortedIndices = ordered.keySet().stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(sortedIndices);
        double[] sortedValues = new double[sortedIndices.length];
        for (int i = 0; i < sortedIndices.length; i++) {
            sortedValues[i] = ordered.get(sortedIndices[i]);
        }
        return new FixedParameters(sortedIndices, sortedValues);
    }

    public static FixedParameters fromNames(Map<String, Double> valuesByName, String[] parameterNames) {
        Objects.requireNonNull(valuesByName, "valuesByName");
        Objects.requireNonNull(parameterNames, "parameterNames");
        if (valuesByName.isEmpty()) {
            return NONE;
        }
        int[] indices = new int[valuesByName.size()];
        double[] values = new double[valuesByName.size()];
        int position = 0;
        for (Map.Entry<String, Double> entry : valuesByName.entrySet()) {
            String name = Objects.requireNonNull(entry.getKey(), "parameter name");
            int index = indexOf(parameterNames, name);
            if (index < 0) {
                throw new IllegalArgumentException("unknown parameter name: " + name);
            }
            indices[position] = index;
            values[position] = Objects.requireNonNull(entry.getValue(), "parameter value");
            position++;
        }
        return of(indices, values);
    }

    public boolean isEmpty() {
        return indices.length == 0;
    }

    public int size() {
        return indices.length;
    }

    public int[] indices() {
        return indices.clone();
    }

    public double[] values() {
        return values.clone();
    }

    public boolean isFixed(int index) {
        return Arrays.binarySearch(indices, index) >= 0;
    }

    public double value(int index) {
        int position = Arrays.binarySearch(indices, index);
        if (position < 0) {
            throw new IllegalArgumentException("parameter is not fixed: " + index);
        }
        return values[position];
    }

    public FixedParameters validate(int parameterCount) {
        if (parameterCount < 0) {
            throw new IllegalArgumentException("parameterCount must be non-negative");
        }
        for (int index : indices) {
            if (index >= parameterCount) {
                throw new IllegalArgumentException("fixed parameter index out of range: " + index);
            }
        }
        return this;
    }

    public boolean[] mask(int parameterCount) {
        validate(parameterCount);
        boolean[] mask = new boolean[parameterCount];
        for (int index : indices) {
            mask[index] = true;
        }
        return mask;
    }

    public int[] freeIndices(int parameterCount) {
        boolean[] fixed = mask(parameterCount);
        int count = 0;
        for (boolean value : fixed) {
            if (!value) {
                count++;
            }
        }
        int[] free = new int[count];
        int position = 0;
        for (int index = 0; index < fixed.length; index++) {
            if (!fixed[index]) {
                free[position++] = index;
            }
        }
        return free;
    }

    public double[] apply(double[] params) {
        double[] copy = params.clone();
        applyInPlace(copy);
        return copy;
    }

    public void applyInPlace(double[] params) {
        Objects.requireNonNull(params, "params");
        validate(params.length);
        for (int i = 0; i < indices.length; i++) {
            params[indices[i]] = values[i];
        }
    }

    private static int indexOf(String[] names, String name) {
        for (int i = 0; i < names.length; i++) {
            if (name.equals(names[i])) {
                return i;
            }
        }
        return -1;
    }
}