package com.neural.mnist;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

public class Loader {
    int BUFFER_SIZE = 8 * 1024;
    int IMAGE_WIDTH = 28;
    int IMAGE_HEIGHT = 28;
    int RESULT_SIZE = 10;
    int RESULT_SIZE_TO_READ = 1;

//    public static void main(String args[]) {
//        Loader loader = new Loader();
//        ClassLoader classLoader = loader.getClass().getClassLoader();
//        File imagesFile = loader.getFile("mnist/train/train-images.idx3-ubyte");
//        File labelsFile = loader.getFile("mnist/train/train-labels.idx1-ubyte");
//        List<NeuralType> trainImages = loader.getData(imagesFile, labelsFile);
//    }

    public List<NeuralType> load(String imagesFilePath, String labelsFilePath, int numberOfExpectedImages){
        File imagesFile = getFile(imagesFilePath);
        File labelsFile = getFile(labelsFilePath);
        List<NeuralType> images = getData(imagesFile, labelsFile, numberOfExpectedImages);
        return images;
    }

    public List<NeuralType> getData(File imagesFile, File labelsFile, int numberOfExpectedImages) {
        List<NeuralType> images = new ArrayList<>();

        try ( InputStream imageStream = new FileInputStream(imagesFile);
              InputStream labelStream = new FileInputStream(labelsFile)) {
            BufferedInputStream bImageStream = new BufferedInputStream(imageStream, BUFFER_SIZE);
            BufferedInputStream bLabelStream = new BufferedInputStream(labelStream, BUFFER_SIZE);

            int imageMagicNumber = readInt32(bImageStream);
            assertThat(imageMagicNumber == 2051);
            int labelMagicNumber = readInt32(bLabelStream);
            assertThat(labelMagicNumber == 2049);

            int numberOfImages = readInt32(bImageStream);
            int numberOfLabels = readInt32(bLabelStream);
            assertThat(numberOfImages == numberOfLabels);

            int numberOfRows = readInt32(bImageStream);
            int numberOfColumns = readInt32(bImageStream);
            assertThat(numberOfRows == numberOfColumns);

            for (int i = 0; i < numberOfImages; i++) {
                images.add(NeuralObjectReader.read(bImageStream, bLabelStream, IMAGE_WIDTH, IMAGE_HEIGHT, RESULT_SIZE, RESULT_SIZE_TO_READ));
            }

            assertThat(numberOfExpectedImages == images.size());

            bImageStream.close();
            bLabelStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return images;
    }

    public int readInt32(BufferedInputStream bis) {
        byte[] byteArray = new byte[4];
        try {
            bis.read(byteArray);
        } catch (IOException e) {
            e.printStackTrace();
        }
        ByteBuffer wrapped = ByteBuffer.wrap(byteArray); // big-endian by default
        return wrapped.getInt();
    }

    public File getFile(String fileName) {
        //Get file from resources folder
        ClassLoader classLoader = getClass().getClassLoader();
        return new File(classLoader.getResource(fileName).getFile());
    }
}
