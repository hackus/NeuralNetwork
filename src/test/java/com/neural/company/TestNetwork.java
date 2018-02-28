package com.neural.company;

import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.core.util.StatusPrinter;
import com.neural.NeuralLayer;
import com.neural.NeuralNetwork;
import com.neural.functions.ActivationFunction;
import com.neural.mnist.Loader;
import com.neural.mnist.NeuralObject;
import com.neural.mnist.NeuralType;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.exp;

public class TestNetwork {

    final static Logger logger = LoggerFactory.getLogger(NeuralNetwork.class);

    @Test
    public void test_small(){
        Random rnd = new Random();
        ActivationFunction<Double> defaultActivationFunction = (value) -> { return value; };
        ActivationFunction<Double> activationFunction = (value) -> {return 1 / (1 + exp(-value));};

        List<NeuralLayer> layers = new ArrayList<>();
        layers.add(new NeuralLayer(2, defaultActivationFunction));
        layers.add(new NeuralLayer(2, activationFunction));
        layers.add(new NeuralLayer(2, activationFunction));

        List<NeuralType> images = buildListOfImages();

        NeuralNetwork network = new NeuralNetwork();
        network.buildNetwork(layers);
        network.learn(images, images,10000);
        network.evaluate(images);
        network.print();
    }

    public List<NeuralType>  buildListOfImages(){
        List<NeuralType>  images = new ArrayList<>();
        NeuralObject img1 = new NeuralObject(1,2,2);

        img1.setInput(Arrays.asList(ArrayUtils.toObject(new double[]{0.05d, 0.1d})));

        img1.setResult(Arrays.asList(ArrayUtils.toObject(new double[]{0.01d, 0.99d})));

        images.add(img1);

        return images;
    }

    @Test
    public void test1(){

        LoggerContext lc = (LoggerContext) LoggerFactory.getILoggerFactory();
        StatusPrinter.print(lc);

        ActivationFunction<Double> defaultActivationFunction = (value) -> { return value/10000; };
        ActivationFunction<Double> activationFunction = (value) -> {return 1 / (1 + exp(-value));};

        Loader loader = new Loader();
        List<NeuralType> trainImages = loader.load("mnist/train/train-images.idx3-ubyte",
                "mnist/train/train-labels.idx1-ubyte", 60000);
        List<NeuralType> testImages = loader.load("mnist/testing/t10k-images.idx3-ubyte",
                "mnist/testing/t10k-labels.idx1-ubyte", 10000);

        List<NeuralLayer> layers = new ArrayList<>();
        layers.add(new NeuralLayer(784, defaultActivationFunction));
        layers.add(new NeuralLayer(30, activationFunction));
        layers.add(new NeuralLayer(10, activationFunction));

        NeuralNetwork network = new NeuralNetwork();
        network.buildNetwork(layers);
        network.learn(trainImages, testImages, 100);
    }
}
