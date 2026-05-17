package com.curioloop.yum4j.stats.tool;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lightweight CSV loader for test fixtures.
 *
 * <p>Mirrors the {@code LoadCsv} helper in {@code reference/tsa/_data_/testing.go}.
 * Reads columnar CSV files (header row + data rows) from the classpath under
 * {@code src/test/resources/statsmodels/} and returns each column as a
 * {@code double[]}.</p>
 */
final class CsvFixtures {

    private CsvFixtures() {}

    /**
     * Loads a columnar CSV file from the classpath.
     *
     * @param resource classpath-relative path (e.g. {@code "statsmodels/datasets/macrodata/macrodata.csv"})
     * @return map from column header to {@code double[]} values
     */
    static Map<String, double[]> loadColumnar(String resource) {
        try (InputStream is = openResource(resource);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {

            String headerLine = br.readLine();
            Objects.requireNonNull(headerLine, "empty resource: " + resource);
            String[] headers = headerLine.split(",");
            for (int i = 0; i < headers.length; i++) {
                headers[i] = headers[i].trim().replaceAll("^\"|\"$", "");
            }

            List<String[]> rows = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.isBlank()) rows.add(line.split(","));
            }

            Map<String, double[]> result = new LinkedHashMap<>();
            for (int col = 0; col < headers.length; col++) {
                double[] values = new double[rows.size()];
                for (int row = 0; row < rows.size(); row++) {
                    String cell = rows.get(row)[col].trim().replaceAll("^\"|\"$", "");
                    values[row] = cell.equals("NA") ? Double.NaN : Double.parseDouble(cell);
                }
                result.put(headers[col], values);
            }
            return result;
        } catch (IOException e) {
            throw new RuntimeException("Cannot load fixture: " + resource, e);
        }
    }

    /**
     * Loads a single-column CSV file (no header) from the classpath.
     *
     * @param resource classpath-relative path
     * @return the values as a {@code double[]}
     */
    static double[] loadSingleColumn(String resource) {
        try (InputStream is = openResource(resource);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {

            br.readLine(); // skip header
            List<Double> values = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.isBlank()) values.add(Double.parseDouble(line.trim()));
            }
            return values.stream().mapToDouble(Double::doubleValue).toArray();
        } catch (IOException e) {
            throw new RuntimeException("Cannot load fixture: " + resource, e);
        }
    }

    private static InputStream openResource(String resource) {
        InputStream is = CsvFixtures.class.getClassLoader().getResourceAsStream(resource);
        if (is == null) {
            throw new RuntimeException("Classpath resource not found: " + resource);
        }
        return is;
    }
}
