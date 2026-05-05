package com.curioloop.yum4j.fixtures;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class StatsmodelsResources {

    private static final String STATESPACE_RESULTS_ROOT = "statsmodels/tsa/statespace/tests/results/";
    private static final String MACRODATA_RESOURCE = "statsmodels/datasets/macrodata/macrodata.csv";

    private StatsmodelsResources() {
    }

    public static List<String[]> readStatespaceResultsCsv(String fileName) {
        return readCsv(STATESPACE_RESULTS_ROOT + fileName);
    }

    public static double[][] readMacrodata() {
        return readNumericCsv(MACRODATA_RESOURCE);
    }

    public static Map<String, double[]> readMacrodataColumns() {
        return readNumericCsvColumns(MACRODATA_RESOURCE);
    }

    public static List<String[]> readCsv(String resourcePath) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(openResource(resourcePath), StandardCharsets.UTF_8))) {
            return reader.lines()
                .map(StatsmodelsResources::splitCsvLine)
                .toList();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read reference csv: " + resourcePath, e);
        }
    }

    public static double[][] readNumericCsv(String resourcePath) {
        List<String[]> rows = readCsv(resourcePath);
        List<double[]> numericRows = new ArrayList<>();
        for (int rowIndex = 1; rowIndex < rows.size(); rowIndex++) {
            numericRows.add(parseDoubles(rows.get(rowIndex)));
        }
        return numericRows.toArray(new double[0][]);
    }

    public static Map<String, double[]> readNumericCsvColumns(String resourcePath) {
        List<String[]> rows = readCsv(resourcePath);
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("CSV has no header: " + resourcePath);
        }
        String[] header = rows.get(0);
        double[][] values = new double[Math.max(0, rows.size() - 1)][];
        for (int rowIndex = 1; rowIndex < rows.size(); rowIndex++) {
            values[rowIndex - 1] = parseDoubles(rows.get(rowIndex));
        }
        Map<String, double[]> columns = new LinkedHashMap<>();
        for (int column = 0; column < header.length; column++) {
            double[] columnValues = new double[values.length];
            for (int row = 0; row < values.length; row++) {
                columnValues[row] = values[row][column];
            }
            columns.put(header[column].trim(), columnValues);
        }
        return columns;
    }

    private static double[] parseDoubles(String[] row) {
        double[] values = new double[row.length];
        for (int i = 0; i < row.length; i++) {
            values[i] = Double.parseDouble(row[i].trim());
        }
        return values;
    }

    private static String[] splitCsvLine(String line) {
        return line.replace("\"", "").split(",", -1);
    }

    private static InputStream openResource(String resourcePath) throws IOException {
        InputStream input = StatsmodelsResources.class.getClassLoader().getResourceAsStream(resourcePath);
        if (input == null) {
            throw new FileNotFoundException(resourcePath);
        }
        return input;
    }
}