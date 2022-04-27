import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    public final double[] output_Bias;
    public final double[] hidden_Bias;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;

    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate, double[] hidden_Bias, double[] output_Bias) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
        this.hidden_Bias = hidden_Bias;
        this.output_Bias = output_Bias;
    }

    //Calculate neuron activation for an input
    public double sigmoid(double input) {
        return 1 / (1 + Math.pow(Math.E, -input)); //TODO!
    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;
            for (int k = 0; k < num_inputs; k++) {
                weighted_sum += hidden_layer_weights[k][i] * inputs[k];
            }
            weighted_sum += hidden_Bias[i];
            output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;
            for (int k = 0; k < num_hidden; k++) {
                weighted_sum += output_layer_weights[k][i] * hidden_layer_outputs[k];
            }
            weighted_sum += output_Bias[i];
            output = sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }
        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs, double[] output_layer_outputs, int[] desired_outputs) {
        // TODO! Calculate output layer betas.
        double[] output_layer_betas = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
//            if(output_layer_outputs[i] == desired_outputs[i]){
//                output_layer_betas[i] = 1 - output_layer_outputs[i];
//            }else{
//                output_layer_betas[i] = 0 - output_layer_outputs[i];
//            }

            output_layer_betas[i] = desired_outputs[i] - output_layer_outputs[i];
        }
        //System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        // TODO! Calculate hidden layer betas.
        double[] hidden_layer_betas = new double[num_hidden];
        for (int j = 0; j < num_hidden; j++) {
            for (int k = 0; k < num_outputs; k++) {
                hidden_layer_betas[j] += output_layer_weights[j][k] * output_layer_outputs[k] * (1 - output_layer_outputs[k]) * output_layer_betas[k];
            }
        }
        // System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // TODO! Calculate output layer Bias.
        double[] delta_output_bias = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            delta_output_bias[i] = learning_rate * output_layer_outputs[i] * (1 - output_layer_outputs[i]) * output_layer_betas[i];
        }
        // TODO! Calculate Hidden layer Bias.
        double[] delta_hidden_bias = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            delta_hidden_bias[i] = learning_rate * hidden_layer_outputs[i] * (1 - hidden_layer_outputs[i]) * hidden_layer_betas[i];
        }

        // Todo! Calculate output layer weight changes. This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        for (int i = 0; i < num_hidden; i++) {
            for (int j = 0; j < num_outputs; j++) {
                delta_output_layer_weights[i][j] = learning_rate * hidden_layer_outputs[i] * output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j];
            }
        }

        // Todo! Calculate hidden layer weight changes. This is a IxH array (I input's, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        for (int i = 0; i < num_inputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                delta_hidden_layer_weights[i][j] = learning_rate * inputs[i] * hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) * hidden_layer_betas[j];
            }
        }

        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights,new double[][]{delta_hidden_bias},new double[][]{delta_output_bias}};
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights, double[][] delta_hidden_bias,double[][] delta_output_bias) {
        // TODO! Update the bias
/*
  for (int i = 0; i < num_hidden; i++){
            this.hidden_Bias[i] = delta_hidden_bias[0][i];
        }

        for (int i = 0; i < num_outputs; i++){
            this.output_Bias[i] = delta_output_bias[0][i];
        }
 */
        for (int i = 0; i < num_hidden; i++){
            this.hidden_Bias[i] += delta_hidden_bias[0][i];
        }

        for (int i = 0; i < num_outputs; i++){
            this.output_Bias[i] += delta_output_bias[0][i];
        }
//        if (num_hidden >= 0) System.arraycopy(delta_hidden_bias[0], 0, this.hidden_Bias, 0, num_hidden);
//
//        if (num_outputs >= 0) System.arraycopy(delta_output_bias[0], 0, this.output_Bias, 0, num_outputs);
        // TODO! Update the weights
        for (int i = 0; i < num_hidden; i++) {
            for (int j = 0; j < num_inputs; j++) {
                hidden_layer_weights[j][i] = delta_hidden_layer_weights[j][i] + hidden_layer_weights[j][i];
            }
        }

        for (int i = 0; i < num_outputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                output_layer_weights[j][i] = delta_output_layer_weights[j][i] + output_layer_weights[j][i];
            }
        }
        //System.out.println("Placeholder");
    }

    public void train(double[][] instances, int[][] desired_outputs, int epochs, int[] integer_encodedDesired) {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            int[] predictions = new int[instances.length];

            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);

                int predicted_class = -1; // TODO!
                double[] output_layer_outputs = outputs[1];
                double highest = 0;

                for (int k = 0; k < output_layer_outputs.length; k++) {
                    if (output_layer_outputs[k] > highest) {
                        highest = output_layer_outputs[k];
                        predicted_class = k;
                    }
                }
                predictions[i] = predicted_class;
//                System.out.println("delta weight: " + Arrays.deepToString(delta_weights[2]));
//                System.out.println("delta weight: " + Arrays.deepToString(delta_weights[3]));
                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1], delta_weights[2], delta_weights[3]);
            }

            // TODO: Print accuracy achieved over this epoch
            int counter = 0;
            for (int i = 0; i <= desired_outputs.length - 1; i++) {
                if (predictions[i] == integer_encodedDesired[i]) {
                    counter++;
                }
            }
            double acc = (double) counter / desired_outputs.length;
            if((epoch % 5) == 0 || epoch == 1)
            System.out.println("epoch = " + epoch + ": Training Data accuracy = " + acc);
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            // TODO !Should be 0, 1, or 2.
            int predicted_class = -1;
            double[] output_layer_outputs = outputs[1];
            double highest = 0;
            for (int k = 0; k < output_layer_outputs.length; k++) {
                if (output_layer_outputs[k] > highest) {
                    highest = output_layer_outputs[k];
                    predicted_class = k;
                }
            }
            if(i == 0) {
                System.out.println("Output_layer_outputs for instance =  " + i + ", is: " + Arrays.toString(output_layer_outputs));
            }
            predictions[i] = predicted_class;
        }
        return predictions;
    }

}
