package com.neural.mnist;

import java.io.BufferedInputStream;
import java.io.IOException;

import static org.assertj.core.api.Assertions.assertThat;

public class NeuralObjectReader {
    public static NeuralObject read(final BufferedInputStream imageStream, final BufferedInputStream labelStream, final int rowSize, final int colSize, final int resultSize, final int resultSizeToRead) throws IOException {
        NeuralObject img = new NeuralObject(rowSize, colSize, resultSize);

        for(int i=0;i<rowSize;i++){
            for(int j=0;j<colSize;j++){
                int value = imageStream.read();
                if(value == -1) throw new RuntimeException("Error reading immage");
                img.putInput(i,j, Double.valueOf(value));
            }
        }

        for(int i=0;i<resultSizeToRead;i++){
            int value = labelStream.read();
            if(value == -1) throw new RuntimeException("Error reading label");
            img.putOutput(value);
        }

        return img;
    }
}
