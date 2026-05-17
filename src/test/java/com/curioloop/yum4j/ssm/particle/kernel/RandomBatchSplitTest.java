package com.curioloop.yum4j.ssm.particle.kernel;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies {@link RandomBatch#split(long)} for per-slab RNG derivation.
 *
 * <p>Key properties:
 * <ul>
 *   <li>Deterministic: same masterSeed + same key → identical stream</li>
 *   <li>Independent: different keys → different streams</li>
 *   <li>Applicable to STRIPED parallel strategy where key = slabIndex</li>
 * </ul>
 *
 * <p><b>Validates: Requirements R10.5</b>
 */
class RandomBatchSplitTest {

    private static final long MASTER_SEED = 0xCAFEBABE_DEADBEEFL;
    private static final int STREAM_LEN = 256;

    @Test
    void splitSameKeyProducesSameStream() {
        RandomBatch master1 = RandomBatch.of(MASTER_SEED);
        RandomBatch master2 = RandomBatch.of(MASTER_SEED);

        RandomBatch child1 = master1.split(42L);
        RandomBatch child2 = master2.split(42L);

        double[] out1 = new double[STREAM_LEN];
        double[] out2 = new double[STREAM_LEN];

        // Verify uniform streams are identical
        child1.nextUniforms(out1, 0, STREAM_LEN);
        child2.nextUniforms(out2, 0, STREAM_LEN);
        assertArrayEquals(out1, out2, "split(42) from same masterSeed must produce identical uniform streams");

        // Verify Gaussian streams are identical (fresh splits)
        RandomBatch g1 = RandomBatch.of(MASTER_SEED).split(42L);
        RandomBatch g2 = RandomBatch.of(MASTER_SEED).split(42L);
        g1.nextGaussians(out1, 0, STREAM_LEN);
        g2.nextGaussians(out2, 0, STREAM_LEN);
        assertArrayEquals(out1, out2, "split(42) from same masterSeed must produce identical Gaussian streams");
    }

    @Test
    void splitDifferentKeysProduceDifferentStreams() {
        RandomBatch master = RandomBatch.of(MASTER_SEED);

        RandomBatch child0 = master.split(0L);
        RandomBatch child1 = master.split(1L);

        double[] out0 = new double[STREAM_LEN];
        double[] out1 = new double[STREAM_LEN];

        child0.nextUniforms(out0, 0, STREAM_LEN);
        child1.nextUniforms(out1, 0, STREAM_LEN);

        // Streams from different keys must differ
        boolean allEqual = true;
        for (int i = 0; i < STREAM_LEN; i++) {
            if (out0[i] != out1[i]) {
                allEqual = false;
                break;
            }
        }
        assertFalse(allEqual, "split(0) and split(1) must produce different streams");
    }

    @Test
    void splitDeterministicAcrossMultipleSlabIndices() {
        // Simulates the STRIPED use case: each slab gets split(slabIndex)
        int slabCount = 8;
        double[][] streams = new double[slabCount][STREAM_LEN];

        // First pass: generate streams for each slab
        for (int slab = 0; slab < slabCount; slab++) {
            RandomBatch child = RandomBatch.of(MASTER_SEED).split(slab);
            child.nextUniforms(streams[slab], 0, STREAM_LEN);
        }

        // Second pass: regenerate and verify determinism
        for (int slab = 0; slab < slabCount; slab++) {
            RandomBatch child = RandomBatch.of(MASTER_SEED).split(slab);
            double[] verify = new double[STREAM_LEN];
            child.nextUniforms(verify, 0, STREAM_LEN);
            assertArrayEquals(streams[slab], verify,
                    "split(" + slab + ") must be deterministic across invocations");
        }

        // Verify all slab streams are pairwise distinct
        for (int i = 0; i < slabCount; i++) {
            for (int j = i + 1; j < slabCount; j++) {
                boolean differ = false;
                for (int k = 0; k < STREAM_LEN; k++) {
                    if (streams[i][k] != streams[j][k]) {
                        differ = true;
                        break;
                    }
                }
                assertTrue(differ, "Slab " + i + " and slab " + j + " must produce different streams");
            }
        }
    }

    @Test
    void splitIsIndependentOfParentStreamConsumption() {
        // split() derives from masterSeed, not from current RNG state.
        // Consuming draws from the parent should not affect the child.
        RandomBatch master1 = RandomBatch.of(MASTER_SEED);
        RandomBatch master2 = RandomBatch.of(MASTER_SEED);

        // Advance master2's internal state
        double[] discard = new double[1000];
        master2.nextUniforms(discard, 0, 1000);
        master2.nextGaussians(discard, 0, 1000);

        // Both splits should produce the same child stream
        RandomBatch child1 = master1.split(7L);
        RandomBatch child2 = master2.split(7L);

        double[] out1 = new double[STREAM_LEN];
        double[] out2 = new double[STREAM_LEN];
        child1.nextUniforms(out1, 0, STREAM_LEN);
        child2.nextUniforms(out2, 0, STREAM_LEN);

        assertArrayEquals(out1, out2,
                "split(key) must depend only on masterSeed, not on parent's consumed state");
    }

    @Test
    void splitExponentialStreamDeterministic() {
        RandomBatch child1 = RandomBatch.of(MASTER_SEED).split(3L);
        RandomBatch child2 = RandomBatch.of(MASTER_SEED).split(3L);

        double[] out1 = new double[STREAM_LEN];
        double[] out2 = new double[STREAM_LEN];

        child1.nextExponentials(out1, 0, STREAM_LEN, 2.5);
        child2.nextExponentials(out2, 0, STREAM_LEN, 2.5);

        assertArrayEquals(out1, out2,
                "split(3) exponential streams must be deterministic");
    }
}
