package model;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.List;

public class TrainedModel {

    private List<String> labels;
    private long testTime;
    private long trainingTime;
    private MultiLayerNetwork network;
    private Evaluation evaluation;

    public Configuration getConfiguration() {
        return configuration;
    }

    public void setConfiguration(Configuration configuration) {
        this.configuration = configuration;
    }

    private Configuration configuration;

    public void setTrainingTime(long trainingTime) {
        this.trainingTime = trainingTime;
    }

    public long getTrainingTime() {
        return trainingTime;
    }

    public void setNetwork(MultiLayerNetwork network) {
        this.network = network;
    }

    public MultiLayerNetwork getNetwork() {
        return network;
    }

    public void setTestTime(long testTime) {
        this.testTime = testTime;
    }

    public long getTestTime() {
        return testTime;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    public List<String> getLabels() {
        return labels;
    }

    @Override
    public String toString() {
        return "TrainedModel{" +
                "labels=" + labels +
                ", testTime=" + testTime +
                ", trainingTime=" + trainingTime +
                ", network=" + network +
                ", configuration=" + configuration +
                '}';
    }

    public void setEvaluation(Evaluation evaluation) {
        this.evaluation = evaluation;
    }

    public Evaluation getEvaluation() {
        return evaluation;
    }
}
