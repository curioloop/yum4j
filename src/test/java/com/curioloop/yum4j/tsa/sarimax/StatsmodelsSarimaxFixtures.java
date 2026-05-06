package com.curioloop.yum4j.tsa.sarimax;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class StatsmodelsSarimaxFixtures {

    private static final String RESULTS_ROOT = "statsmodels/tsa/statespace/tests/results/";
    private static final String SARIMAX_RESULTS = readReferenceResource("results_sarimax.py");

    private StatsmodelsSarimaxFixtures() {
    }

    static double[] array(String variableName) {
        return parseArray(resolveArrayLiteral(assignmentValue(variableName)));
    }

    static double[] dictArray(String dictName, String key) {
        return parseArray(resolveArrayLiteral(dictValue(dictName, key)));
    }

    static double dictScalar(String dictName, String key) {
        return parseNumberExpression(dictValue(dictName, key));
    }

    static List<String> coverageLines() {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(openResource("results_sarimax_coverage.csv"), StandardCharsets.UTF_8))) {
            return reader.lines().toList();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read statsmodels SARIMAX coverage reference", e);
        }
    }

    private static String readReferenceResource(String fileName) {
        try (InputStream input = openResource(fileName)) {
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read statsmodels SARIMAX reference: " + fileName, e);
        }
    }

    private static InputStream openResource(String fileName) throws IOException {
        String resourcePath = RESULTS_ROOT + fileName;
        InputStream input = StatsmodelsSarimaxFixtures.class.getClassLoader().getResourceAsStream(resourcePath);
        if (input == null) {
            throw new FileNotFoundException(resourcePath);
        }
        return input;
    }

    private static String assignmentValue(String variableName) {
        Pattern pattern = Pattern.compile("(?m)^" + Pattern.quote(variableName) + "\\s*=\\s*");
        Matcher matcher = pattern.matcher(SARIMAX_RESULTS);
        if (!matcher.find()) {
            throw new IllegalArgumentException("Missing statsmodels SARIMAX fixture variable: " + variableName);
        }
        ValueSpan span = readValue(SARIMAX_RESULTS, matcher.end());
        return SARIMAX_RESULTS.substring(span.start(), span.end()).trim();
    }

    private static String dictValue(String dictName, String key) {
        String dict = assignmentValue(dictName);
        if (!dict.startsWith("{") || !dict.endsWith("}")) {
            throw new IllegalArgumentException("Statsmodels fixture is not a dict: " + dictName);
        }
        int index = 1;
        while (index < dict.length() - 1) {
            index = skipSeparators(dict, index);
            if (index >= dict.length() - 1) {
                break;
            }
            String entryKey = readQuotedString(dict, index);
            index = skipWhitespace(dict, index + entryKey.length() + 2);
            if (index >= dict.length() || dict.charAt(index) != ':') {
                throw new IllegalArgumentException("Malformed statsmodels fixture dict: " + dictName);
            }
            ValueSpan span = readValue(dict, index + 1);
            if (entryKey.equals(key)) {
                return dict.substring(span.start(), span.end()).trim();
            }
            index = span.end();
        }
        throw new IllegalArgumentException("Missing key " + key + " in statsmodels fixture dict: " + dictName);
    }

    private static String resolveArrayLiteral(String value) {
        String trimmed = value.trim();
        if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
            return trimmed;
        }
        if (trimmed.matches("[A-Za-z_][A-Za-z0-9_]*")) {
            return resolveArrayLiteral(assignmentValue(trimmed));
        }
        throw new IllegalArgumentException("Statsmodels fixture value is not an array: " + value);
    }

    private static double[] parseArray(String literal) {
        String body = literal.substring(1, literal.length() - 1).trim();
        if (body.isEmpty()) {
            return new double[0];
        }
        List<Double> values = new ArrayList<>();
        int start = 0;
        for (int index = 0; index <= body.length(); index++) {
            if (index == body.length() || body.charAt(index) == ',') {
                String part = body.substring(start, index).trim();
                if (!part.isEmpty()) {
                    values.add(parseNumberExpression(part));
                }
                start = index + 1;
            }
        }
        double[] result = new double[values.size()];
        for (int index = 0; index < values.size(); index++) {
            result[index] = values.get(index);
        }
        return result;
    }

    private static double parseNumberExpression(String expression) {
        String value = expression.trim();
        int power = value.indexOf("**");
        if (power >= 0) {
            double base = parseNumberExpression(value.substring(0, power));
            double exponent = parseNumberExpression(value.substring(power + 2));
            return Math.pow(base, exponent);
        }
        return Double.parseDouble(value);
    }

    private static ValueSpan readValue(String source, int from) {
        int start = skipWhitespace(source, from);
        char first = source.charAt(start);
        if (first == '[' || first == '{') {
            return new ValueSpan(start, findMatchingBracket(source, start) + 1);
        }
        int end = start;
        while (end < source.length()) {
            char ch = source.charAt(end);
            if (ch == ',' || ch == '\n' || ch == '}') {
                break;
            }
            end++;
        }
        return new ValueSpan(start, end);
    }

    private static int findMatchingBracket(String source, int openIndex) {
        char open = source.charAt(openIndex);
        char close = open == '[' ? ']' : '}';
        int depth = 0;
        char quote = 0;
        for (int index = openIndex; index < source.length(); index++) {
            char ch = source.charAt(index);
            if ((ch == '"' || ch == '\'') && (index == 0 || source.charAt(index - 1) != '\\')) {
                quote = quote == ch ? 0 : quote == 0 ? ch : quote;
            }
            if (quote != 0) {
                continue;
            }
            if (ch == open) {
                depth++;
            } else if (ch == close) {
                depth--;
                if (depth == 0) {
                    return index;
                }
            }
        }
        throw new IllegalArgumentException("Unclosed statsmodels fixture literal starting at " + openIndex);
    }

    private static String readQuotedString(String source, int from) {
        int start = skipWhitespace(source, from);
        char quote = source.charAt(start);
        if (quote != '"' && quote != '\'') {
            throw new IllegalArgumentException("Expected quoted statsmodels fixture key at " + start);
        }
        int end = source.indexOf(quote, start + 1);
        if (end < 0) {
            throw new IllegalArgumentException("Unclosed statsmodels fixture key at " + start);
        }
        return source.substring(start + 1, end);
    }

    private static int skipSeparators(String source, int from) {
        int index = from;
        while (index < source.length()) {
            char ch = source.charAt(index);
            if (!Character.isWhitespace(ch) && ch != ',') {
                break;
            }
            index++;
        }
        return index;
    }

    private static int skipWhitespace(String source, int from) {
        int index = from;
        while (index < source.length() && Character.isWhitespace(source.charAt(index))) {
            index++;
        }
        return index;
    }

    private record ValueSpan(int start, int end) {
    }
}