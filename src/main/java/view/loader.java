package view;


import controller.Trainer;
import model.Configuration;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class loader {

    public static void main(String args[]){
        // train model
Configuration configuration = new Configuration();
        configuration.setBatchSize(18);
        configuration.setChannels(3);
        configuration.setEpochs(50);
        configuration.setFilePath("C:\\Users\\okesa\\Documents\\Deep learning\\deeplearning4j\\findme\\data");
        configuration.setHeight(100);
        configuration.setMaxNumberOfData(100);
        configuration.setNumberOfCores(8);
        configuration.setNumberOfIterations(1);
        configuration.setNumberOfLabels(2);
        configuration.setSeed(42);
        configuration.setWidth(100);

        Trainer trainer = new Trainer(configuration);
        try {
            trainer.train();
            System.out.println(trainer.getModel().toString());

            File file = new File("C:\\Users\\okesa\\Documents\\Deep learning\\deeplearning4j\\findme\\passport.png");
            System.out.println( Trainer.predict(trainer.getModel(), file));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
