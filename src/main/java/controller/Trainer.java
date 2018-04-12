package controller;

import model.Configuration;
import model.TrainedModel;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class Trainer {

    private Configuration parameter;
    private Logger log;
    private long startTime;
    private long endTime;
private MultiLayerNetwork network;

    public TrainedModel getModel() {
        return model;
    }

    public void setModel(TrainedModel model) {
        this.model = model;
    }

    private TrainedModel model;

    public Trainer(Configuration params){
        parameter = params;
        log  = LoggerFactory.getLogger(Trainer.class);
        model = new TrainedModel();
    }

    /**
     *
     * Trains a network using convolution neural network
     * @return TrainedModel
     * @throws IOException
     */
    public TrainedModel train() throws IOException {

        // check if file path is not null
        if(parameter != null){
            if(parameter.getFilePath() != null){
                // get data file
                File data = new File(parameter.getFilePath());
                Random random = new Random(parameter.getSeed());

                // extract label from parent path
                log.info("Extract label from path");
                ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

                // split data into training and test data
                log.info("split data into training and test data");

                FileSplit fileSplit = new FileSplit(data, new String[] {"jpg","jpeg","png"}, random);
                RandomPathFilter pathFilter = new RandomPathFilter(random, new String[]{"jpg","jpeg","png"}, parameter.getMaxNumberOfData());
                InputSplit[] inputSplits = fileSplit.sample(pathFilter, 0.8, 0.2);
                InputSplit trainData = inputSplits[0];
                InputSplit testData = inputSplits[1];

                ImageRecordReader recordReader = new ImageRecordReader(parameter.getHeight(),parameter.getWidth(),parameter.getChannels(), labelMaker);
recordReader.initialize(trainData);
                List<String> labelNames = recordReader.getLabels();

                DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, parameter.getBatchSize(),1, parameter.getNumberOfLabels());

                // scale pixel values to 0 - 1
                log.info("scale pixel values to 0 - 1");
                DataNormalization scaler = new ImagePreProcessingScaler(0,1);

                scaler.fit(dataSetIterator);
                dataSetIterator.setPreProcessor(scaler);


                // build network
                log.info("build network");
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(parameter.getSeed())
                        .iterations(parameter.getNumberOfIterations())
                        .regularization(false)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .l2(0.005)
                        .learningRate(0.0001)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.RMSPROP).momentum(0.9)
                        .list()
                        .layer(0, convInit("cnn1", parameter.getChannels(), 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                        .layer(1, maxPool("maxpool1", new int[]{2,2}))
                        .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                        .layer(3, maxPool("maxool2", new int[]{2,2}))
                        .layer(4, new DenseLayer.Builder()
                                .nOut(500)
                                .build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(parameter.getNumberOfLabels())
                                .activation(Activation.SOFTMAX)
                                .build())
                        .pretrain(false).backprop(true)
                        .setInputType(InputType.convolutional(parameter.getHeight(), parameter.getWidth(), parameter.getChannels()))
                        .build();

                network =  new MultiLayerNetwork(conf);
                MultipleEpochsIterator iterator  = new MultipleEpochsIterator(parameter.getEpochs(), dataSetIterator, parameter.getNumberOfCores());
                network.setListeners(new ScoreIterationListener(10));
                // measure time taken to train network
                startTime = System.currentTimeMillis();
                network.fit(iterator);
                endTime = System.currentTimeMillis();
                long trainingTime = endTime - startTime;


                // evaluate model
                log.info("evaluate model");
                recordReader.initialize(testData);
                dataSetIterator = new RecordReaderDataSetIterator(recordReader,parameter.getBatchSize(),1, parameter.getNumberOfLabels());
                scaler.fit(dataSetIterator);
                dataSetIterator.setPreProcessor(scaler);

                startTime = System.currentTimeMillis();
                Evaluation evaluation = network.evaluate(dataSetIterator);
                endTime = System.currentTimeMillis();
                long testTime = endTime - startTime;

                System.out.println(evaluation.stats(true));
                model.setEvaluation(evaluation);
                model.setTrainingTime(trainingTime);
                model.setTestTime(testTime);
                model.setLabels(labelNames);
                model.setNetwork(network);
                model.setConfiguration(parameter);


                return model;

            }else{
                log.error("File path was not specified. ");
            }
        }
        return null;
    }
    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    /**
     * Predicts a label using a trained network
     * @param model
     * @param imageFile
     * @return
     * @throws Exception
     */
    public static INDArray predict(TrainedModel model, File imageFile) throws Exception {
        NativeImageLoader imageLoader = new NativeImageLoader(model.getConfiguration().getHeight(), model.getConfiguration().getWidth(), model.getConfiguration().getChannels());
        INDArray image = imageLoader.asMatrix(imageFile);
        return model.getNetwork().output(image);
    }

    /**
     * Saves the model to the specified path.
     * @param path
     */
    public void saveModel(String path) throws IOException{
        File file = new File(path);

        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(network, file, true);
    }
}
