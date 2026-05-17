package com.curioloop.yum4j.ssm.kalman;

import com.curioloop.yum4j.ssm.kalman.filter.KalmanEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SimulationSmootherEngine;
import com.curioloop.yum4j.ssm.kalman.smooth.SmootherEngine;

/** Reusable workspace for the high-level Kalman facade. */
public final class KalmanWorkspace implements AutoCloseable {
    private KalmanEngine.Workspace filterWorkspace;
    private SmootherEngine.Workspace smootherWorkspace;
    private SimulationSmootherEngine.Workspace simulationWorkspace;

    KalmanWorkspace() {
    }

    KalmanEngine.Workspace filterWorkspace() {
        if (filterWorkspace == null) {
            filterWorkspace = KalmanEngine.workspace();
        }
        return filterWorkspace;
    }

    SmootherEngine.Workspace smootherWorkspace() {
        if (smootherWorkspace == null) {
            smootherWorkspace = SmootherEngine.workspace(filterWorkspace());
        }
        return smootherWorkspace;
    }

    SimulationSmootherEngine.Workspace simulationWorkspace() {
        if (simulationWorkspace == null) {
            simulationWorkspace = SimulationSmootherEngine.workspace();
        }
        return simulationWorkspace;
    }

    public void retainReusableSimulationWorkspaces() {
        simulationWorkspace().retainReusableWorkspaces();
    }

    public void retainCfaPosteriorWorkspace() {
        simulationWorkspace().retainCfaPosteriorWorkspace();
    }

    public void trimReusableSimulationWorkspaces() {
        if (simulationWorkspace != null) {
            simulationWorkspace.trimReusableWorkspaces();
        }
    }

    public void releaseRetainedScratch() {
        if (smootherWorkspace != null) {
            smootherWorkspace.releaseRetainedScratch();
        } else if (filterWorkspace != null) {
            filterWorkspace.releaseRetainedScratch();
        }
        if (simulationWorkspace != null) {
            simulationWorkspace.releaseRetainedScratch();
            simulationWorkspace.releaseComplexRetainedScratch();
        }
    }

    public void releaseRetainedResults() {
        if (smootherWorkspace != null) {
            smootherWorkspace.releaseRetainedResults();
        } else if (filterWorkspace != null) {
            filterWorkspace.releaseRetainedResults();
        }
        if (simulationWorkspace != null) {
            simulationWorkspace.releaseRetainedResults();
            simulationWorkspace.releaseComplexRetainedResults();
        }
    }

    @Override
    public void close() {
        releaseRetainedResults();
        releaseRetainedScratch();
    }
}