package com.curioloop.yum4j.stats.test;

import com.curioloop.yum4j.linalg.Regressor;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class StatsmodelsTestData {

    private StatsmodelsTestData() {}

    record DiagnosticFixture(double[] y, double[] exog, double[] residual) {}

    static Map<String, double[]> macrodata() {
        return loadColumnarResource("statsmodels/datasets/macrodata/macrodata.csv", ',');
    }

    static Map<String, double[]> referenceDataset(String path, char delimiter) {
        String resource = "statsmodels/datasets/" + path;
        if (hasResource(resource)) {
            return loadColumnarResource(resource, delimiter);
        }
        return loadColumnarPath(referenceDatasetPath(path), delimiter);
    }

    static DiagnosticFixture diagnosticFixture() {
        Map<String, double[]> macro = macrodata();
        double[] realinv = macro.get("realinv");
        double[] realgdp = macro.get("realgdp");
        double[] realint = macro.get("realint");
        double[] gsRealinv = logDiff400(realinv);
        double[] gsRealgdp = logDiff400(realgdp);
        int n = gsRealinv.length;
        int k = 3;
        double[] y = gsRealinv.clone();
        double[] x = new double[n * k];
        for (int i = 0; i < n; i++) {
            x[i * k] = 1.0;
            x[i * k + 1] = gsRealgdp[i];
            x[i * k + 2] = realint[i];
        }
        var ols = Regressor.ols(y, x.clone(), n, k, Regressor.Opts.QR, Regressor.Opts.INFERENCE);
        return new DiagnosticFixture(y, x, ols.residual(false));
    }

    static double[] logDiff400(double[] x) {
        double[] out = new double[x.length - 1];
        for (int i = 0; i < out.length; i++) out[i] = 400.0 * Math.log(x[i + 1] / x[i]);
        return out;
    }

    static double[] logDiff(double[] x) {
        double[] out = new double[x.length - 1];
        for (int i = 0; i < out.length; i++) out[i] = Math.log(x[i + 1]) - Math.log(x[i]);
        return out;
    }

    static double[] copyOfRange(double[] x, int to) {
        double[] out = new double[to];
        System.arraycopy(x, 0, out, 0, to);
        return out;
    }

    static double[] twoColumn(double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("column lengths differ");
        double[] out = new double[a.length * 2];
        for (int i = 0; i < a.length; i++) {
            out[i * 2] = a[i];
            out[i * 2 + 1] = b[i];
        }
        return out;
    }

    static double[] column(double[] x, int n, int k, int column) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = x[i * k + column];
        return out;
    }

    private static Map<String, double[]> loadColumnarResource(String resource, char delimiter) {
        try (InputStream stream = StatsmodelsTestData.class.getClassLoader().getResourceAsStream(resource)) {
            if (stream == null) throw new IllegalArgumentException("resource not found: " + resource);
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                return loadColumnar(reader, delimiter);
            }
        } catch (IOException e) {
            throw new RuntimeException("cannot load resource: " + resource, e);
        }
    }

    private static Path referenceDatasetPath(String path) {
        return Path.of("reference", "statsmodels", "statsmodels", "datasets", path);
    }

    private static boolean hasResource(String resource) {
        try (InputStream stream = StatsmodelsTestData.class.getClassLoader().getResourceAsStream(resource)) {
            return stream != null;
        } catch (IOException e) {
            return false;
        }
    }

    private static Map<String, double[]> loadColumnarPath(Path path, char delimiter) {
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            return loadColumnar(reader, delimiter);
        } catch (IOException e) {
            throw new RuntimeException("cannot load file: " + path, e);
        }
    }

    private static Map<String, double[]> loadColumnar(BufferedReader reader, char delimiter) throws IOException {
        String headerLine = Objects.requireNonNull(reader.readLine(), "empty csv");
        String[] headers = split(headerLine, delimiter);
        List<String[]> rows = new ArrayList<>();
        String line;
        while ((line = reader.readLine()) != null) {
            if (!line.isBlank()) rows.add(split(line, delimiter));
        }
        Map<String, double[]> result = new LinkedHashMap<>();
        for (int col = 0; col < headers.length; col++) {
            double[] values = new double[rows.size()];
            for (int row = 0; row < rows.size(); row++) {
                String cell = rows.get(row)[col].trim();
                values[row] = cell.equals("NA") ? Double.NaN : Double.parseDouble(cell);
            }
            result.put(headers[col], values);
        }
        return result;
    }

    private static String[] split(String line, char delimiter) {
        String[] parts = line.split(java.util.regex.Pattern.quote(String.valueOf(delimiter)), -1);
        for (int i = 0; i < parts.length; i++) parts[i] = stripQuotes(parts[i].trim());
        return parts;
    }

    private static String stripQuotes(String value) {
        if (value.length() >= 2 && value.charAt(0) == '"' && value.charAt(value.length() - 1) == '"') {
            return value.substring(1, value.length() - 1);
        }
        return value;
    }
}
