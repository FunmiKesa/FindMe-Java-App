package model;

public class Configuration {
    private int maxNumberOfData;
    private int numberOfCores;
    private int epochs;
    private int numberOfIterations;

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getChannels() {
        return channels;
    }

    public void setChannels(int channels) {
        this.channels = channels;
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getNumberOfLabels() {
        return numberOfLabels;
    }

    public void setNumberOfLabels(int numberOfLabels) {
        this.numberOfLabels = numberOfLabels;
    }
    public String getFilePath() {
        return filePath;
    }

    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

    protected int height ;
    protected int width ;
    protected  int channels;
    protected int seed;
    protected int batchSize;
    protected int numberOfLabels;



    protected String filePath;

    @Override
    public String toString() {
        return "Configuration{" +
                "height=" + height +
                ", width=" + width +
                ", channels=" + channels +
                ", seed=" + seed +
                ", batchSize=" + batchSize +
                ", numberOfLabels=" + numberOfLabels +
                ", filePath='" + filePath + '\'' +
                '}';
    }

    public int getMaxNumberOfData() {
        return maxNumberOfData;
    }

    public void setMaxNumberOfData(int maxNumberOfData) {
        this.maxNumberOfData = maxNumberOfData;
    }

    public int getNumberOfCores() {
        return numberOfCores;
    }

    public void setNumberOfCores(int numberOfCores) {
        this.numberOfCores = numberOfCores;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public int getNumberOfIterations() {
        return numberOfIterations;
    }

    public void setNumberOfIterations(int numberOfIterations) {
        this.numberOfIterations = numberOfIterations;
    }
}
