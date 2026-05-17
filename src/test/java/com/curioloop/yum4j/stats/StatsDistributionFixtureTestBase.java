package com.curioloop.yum4j.stats;

import com.curioloop.yum4j.test.RecordedRandomGenerator;
import com.fasterxml.jackson.core.json.JsonReadFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import org.assertj.core.data.Offset;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

abstract class StatsDistributionFixtureTestBase {

    private static final ObjectMapper MAPPER = JsonMapper.builder()
        .enable(JsonReadFeature.ALLOW_NON_NUMERIC_NUMBERS)
        .build();

    protected static JsonNode loadFixture(String fixtureName) {
        String classpath = "/stats/distributions/" + fixtureName;
        try (InputStream in = StatsDistributionFixtureTestBase.class.getResourceAsStream(classpath)) {
            if (in == null) {
                throw new IllegalStateException("Missing fixture on classpath: " + classpath);
            }
            return MAPPER.readTree(in);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read fixture " + classpath, e);
        }
    }

    protected static JsonNode loadFixture(String fixtureName, String sectionName) {
        JsonNode fixture = loadFixture(fixtureName);
        JsonNode section = fixture.get(sectionName);
        if (section == null || section.isNull()) {
            throw new IllegalStateException(
                "Missing section " + sectionName + " in distribution fixture " + fixtureName);
        }
        return section;
    }

    protected static RecordedRandomGenerator replayRng(JsonNode fixture) {
        JsonNode field = fixture.get("_recordedRng");
        if (field == null || field.isNull()) {
            throw new IllegalStateException(
                "Fixture has no _recordedRng entry: " + fixture.get("_fixtureName"));
        }
        if (!field.isArray()) {
            throw new IllegalStateException(
                "Fixture _recordedRng is not an array: " + field.getNodeType());
        }
        return new RecordedRandomGenerator(asDoubleArray(field));
    }

    protected static double[] asDoubleArray(JsonNode array) {
        if (!array.isArray()) {
            throw new IllegalArgumentException(
                "Expected a JSON array; got " + array.getNodeType());
        }
        double[] out = new double[array.size()];
        for (int i = 0; i < out.length; i++) {
            JsonNode element = array.get(i);
            if (element.isNumber()) out[i] = element.doubleValue();
            else if (element.isTextual() && "NaN".equals(element.asText())) out[i] = Double.NaN;
            else if (element.isTextual() && "Infinity".equals(element.asText())) out[i] = Double.POSITIVE_INFINITY;
            else if (element.isTextual() && "-Infinity".equals(element.asText())) out[i] = Double.NEGATIVE_INFINITY;
            else throw new IllegalArgumentException(
                "Non-numeric element at index " + i + ": " + element);
        }
        return out;
    }

    protected static int[] asIntArray(JsonNode array) {
        if (!array.isArray()) {
            throw new IllegalArgumentException(
                "Expected a JSON array; got " + array.getNodeType());
        }
        int[] out = new int[array.size()];
        for (int i = 0; i < out.length; i++) out[i] = array.get(i).intValue();
        return out;
    }

    protected static void assertVectorsClose(double[] expected, double[] actual, Offset<Double> offset) {
        assertThat(actual.length).as("vector length").isEqualTo(expected.length);
        List<Integer> mismatches = new ArrayList<>();
        for (int i = 0; i < expected.length; i++) {
            double expectedValue = expected[i];
            double actualValue = actual[i];
            if (isNonFinite(expectedValue) || isNonFinite(actualValue)) {
                if (Double.doubleToRawLongBits(expectedValue) != Double.doubleToRawLongBits(actualValue)) {
                    mismatches.add(i);
                }
            } else if (Math.abs(expectedValue - actualValue) > offset.value) {
                mismatches.add(i);
            }
        }
        if (!mismatches.isEmpty()) {
            int shown = Math.min(5, mismatches.size());
            StringBuilder sb = new StringBuilder();
            sb.append(mismatches.size()).append(" mismatch(es); first ").append(shown).append(":\n");
            for (int k = 0; k < shown; k++) {
                int i = mismatches.get(k);
                sb.append("  [").append(i).append("] expected=").append(expected[i])
                  .append(" actual=").append(actual[i])
                  .append(" diff=").append(actual[i] - expected[i])
                  .append('\n');
            }
            throw new AssertionError(sb.toString());
        }
    }

    private static boolean isNonFinite(double value) {
        return Double.isNaN(value) || Double.isInfinite(value);
    }
}