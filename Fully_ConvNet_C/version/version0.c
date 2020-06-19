#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define TRUE 1
#define FALSE 0

//Structures-------------------------------------------------------------------------------

typedef struct image
{

    int shape[3]; //[Channel, width, height]
    double ***data;
    int label;

} img;

typedef struct conv_layer
{

    int input_shape[3]; //[Channel, width, height]
    int output_shape[3]; //[Channel, width, height]
    int filter_size[4]; //[filter_width, filter_height]

    int stride_width;
    int stride_height;

    double ****filter; //filter parameters: In_Channel * Out_Channel * width * height
    double *bias; //bais parameters: Out_Channel
    double ****filter_grad;
    double *bias_grad;

    double ***output_grad;

} conv;

typedef struct softmax_layer
{

    int input_dim;
    int output_dim;

    double **weight;
    double *bias;
    double **weight_grad;
    double *bias_grad;
    double *output_grad;

} softmax;

//Function declaration---------------------------------------------------------------------------------------------------------------
img *load_img(int width, int height, FILE *img_file, FILE *label_file, int ind);
void print_img(img *data);
void conv_init(conv *layer, int *prev_dim, int output_channel, int width, int height, int stride_width, int stride_height);
void softmax_init(softmax *layer, conv *prev_layer, int output_dim);
double convolution(conv *layer, double ***input, int current_ch, int w_offset, int h_offset);
double ***conv_forward(conv *layer, double ***input);
double *output(conv *prev_layer, softmax *out, double ***input, short backward);
double leaky_relu(double val);
double loss(double *pred, int label, softmax *out, short backward);
void update(conv **params, softmax *out, double ****forwards, int depth, double learning_rate, double l2_penalty);
double train(conv **params, softmax *out, int depth, img *input, double learning_rate, double l2_penalty);
int inference(conv **params, softmax *out, int depth, img *input, short print);

//Function definition---------------------------------------------------------------------------------

img *load_img(int width, int height, FILE *img_file, FILE *label_file, int ind)
{

    img *data = (img *)malloc(sizeof(img));
    data->shape[0] = 1;
    data->shape[1] = width;
    data->shape[2] = height;

    int w, h;
    unsigned char *one_byte = (unsigned char *)calloc(1, sizeof(unsigned char));

    //Loading Pixel
    data->data = (double ***)calloc(1, sizeof(double **));
    data->data[0] = (double **)calloc(width, sizeof(double *));

    for (w = 0; w < width; w++) {
        data->data[0][w] = (double *)calloc(height, sizeof(double));
        for (h = 0; h < height; h++) {
            fseek(img_file, 16 + (ind * 784) + (28 * w) + h, SEEK_SET);
            fscanf(img_file, "%c", one_byte);
            data->data[0][w][h] = (double)*one_byte / 256;
        }
    }

    //Loading label
    fseek(label_file, 8 + ind, SEEK_SET);
    fscanf(label_file, "%c", one_byte);
    data->label = (int)*one_byte;
    return data;
}

//Print one image at console

void print_img(img *data)
{
    int ch, w, h;

    printf("Label: %d\n", data->label);
    printf(" --------------------------------------------------------\n");

    for (ch = 0; ch < data->shape[0]; ch++) {
        for (w = 0; w < data->shape[1]; w++) {
            printf("|");
            for (h = 0; h < data->shape[2]; h++) {
                if (data->data[ch][w][h] > 0.5) {
                    printf("oo");
                } else {
                    printf("  ");
                }
            }
            printf("|");
            printf("\n");
        }
    }
    printf(" --------------------------------------------------------\n");
}

//Initialize parameters in convolution layers to make training faster and better

void conv_init(conv *layer, int *prev_dim, int output_channel,

               int width, int height, int stride_width, int stride_height) {
    int ind, in_ch, out_ch, fil_w, fil_h, out_width, out_height;
    for (ind = 0; ind < 3; ind++)
    {
        layer->input_shape[ind] = prev_dim[ind];
    }
    layer->output_shape[0] = output_channel;
    layer->output_shape[1] = layer->input_shape[1] / (1 + stride_width);
    layer->output_shape[2] = layer->input_shape[2] / (1 + stride_width);

    layer->filter_size[0] = layer->input_shape[0];
    layer->filter_size[1] = output_channel;
    layer->filter_size[2] = width;
    layer->filter_size[3] = height;

    layer->stride_width = stride_width;
    layer->stride_height = stride_height;

    double he_const = sqrt(6.0 / (double)prev_dim[0]);

    //Filter Weight Initialization
    layer->filter = (double ****)calloc(layer->input_shape[0], sizeof(double ***));
    layer->filter_grad = (double ****)calloc(layer->input_shape[0], sizeof(double ***));
    for (in_ch = 0; in_ch < layer->input_shape[0]; in_ch++) {
        layer->filter[in_ch] = (double ***)calloc(layer->output_shape[0], sizeof(double **));
        layer->filter_grad[in_ch] = (double ***)calloc(layer->output_shape[0], sizeof(double **));
        for (out_ch = 0; out_ch < layer->output_shape[0]; out_ch++) {
            layer->filter[in_ch][out_ch] = (double **)calloc(layer->filter_size[2], sizeof(double *));
            layer->filter_grad[in_ch][out_ch] = (double **)calloc(layer->filter_size[2], sizeof(double *));
            for (fil_w = 0; fil_w < layer->filter_size[2]; fil_w++) {
                layer->filter[in_ch][out_ch][fil_w] = (double *)calloc(layer->filter_size[3], sizeof(double));
                layer->filter_grad[in_ch][out_ch][fil_w] = (double *)calloc(layer->filter_size[3], sizeof(double));
                for (fil_h = 0; fil_h < layer->filter_size[3]; fil_h++) {
                    layer->filter[in_ch][out_ch][fil_w][fil_h] = 2.0 * he_const * rand() / (double)0x7fffffff - he_const;
                    layer->filter_grad[in_ch][out_ch][fil_w][fil_h] = 0.0;
                }
            }
        }
    }

    //Filter Bias Initialization
    layer->bias = (double *)calloc(layer->output_shape[0], sizeof(double));
    layer->bias_grad = (double *)calloc(layer->output_shape[0], sizeof(double));
    for (out_ch = 0; out_ch < layer->output_shape[0]; out_ch++) {
        layer->bias[out_ch] = 0.0;
        layer->bias_grad[out_ch] = 0.0;
    }

    //Output Gradient Initializaition
    layer->output_grad = (double ***)calloc(layer->output_shape[0], sizeof(double **));
    for (out_ch = 0; out_ch < layer->output_shape[0]; out_ch++) {
        layer->output_grad[out_ch] = (double **)calloc(layer->output_shape[1], sizeof(double *));
            for (out_width = 0; out_width < layer->output_shape[1]; out_width++) {
            layer->output_grad[out_ch][out_width] = (double *)calloc(layer->output_shape[2], sizeof(double));
            for (out_height = 0; out_height < layer->output_shape[2]; out_height++) {
                layer->output_grad[out_ch][out_width][out_height] = 0.0;
            }
        }
    }
}

//Initialize parameters in softmax layers to make training faster and better
void softmax_init(softmax *out, conv *prev_layer, int output_dim) {
    
    int in_ch, out_ch;
    out->input_dim = prev_layer->output_shape[0];
    out->output_dim = output_dim;
    double he_const = sqrt(6.0 / (double)prev_layer->output_shape[0]);

    //Weight Initialization
    out->weight = (double **)calloc(out->input_dim, sizeof(double *));
    out->weight_grad = (double **)calloc(out->input_dim, sizeof(double *));
    for (in_ch = 0; in_ch < out->input_dim; in_ch++) {
        out->weight[in_ch] = (double *)calloc(output_dim, sizeof(double));
        out->weight_grad[in_ch] = (double *)calloc(output_dim, sizeof(double));
        for (out_ch = 0; out_ch < output_dim; out_ch++) {
            out->weight[in_ch][out_ch] = 2.0 * he_const * rand() / (double)0x7fffffff - he_const;
            out->weight_grad[in_ch][out_ch] = 0.0;
        }
    }

    //Bias Initialization
    out->bias = (double *)calloc(output_dim, sizeof(double));
    out->bias_grad = (double *)calloc(output_dim, sizeof(double));
    for (out_ch = 0; out_ch < output_dim; out_ch++) {
        out->bias[out_ch] = 0.0;
        out->bias_grad[out_ch] = 0.0;
    }

    //Output Gradient Initializaition
    out->output_grad = (double *)calloc(output_dim, sizeof(double));
    for (out_ch = 0; out_ch < output_dim; out_ch++) {
        out->output_grad[out_ch] = 0.0;
    }
}

//Calculate Convolution
double convolution(conv *layer, double ***input, int current_ch, int w_offset, int h_offset) {

    int in_ch, fil_w, fil_h, pos_w, pos_h;
    double result = 0.0;

    //calculation convolution
    for (in_ch = 0; in_ch < layer->input_shape[0]; in_ch++) {
        for (fil_w = -layer->filter_size[2] / 2; fil_w <= layer->filter_size[2] / 2; fil_w++) {
            for (fil_h = -layer->filter_size[3] / 2; fil_h <= layer->filter_size[3] / 2; fil_h++) {
                //convolution considering padding
                pos_w = w_offset * (1 + layer->stride_width) + fil_w;
                pos_h = h_offset * (1 + layer->stride_height) + fil_h;
                if ((0 <= pos_w) && (pos_w < layer->input_shape[1]) &&
                    (0 <= pos_h) && (pos_h < layer->input_shape[2])) {
                    result += input[in_ch][pos_w][pos_h]
                              * layer->filter[in_ch][current_ch][fil_w + layer->filter_size[2] / 2][fil_h + layer->filter_size[3] / 2];
                }
            }
        }
    }

    result += layer->bias[current_ch];
    result = leaky_relu(result);
    //calculating gradients

    if (result > 0) {
        layer->output_grad[current_ch][w_offset][h_offset] = 1.0;
    } else {
        layer->output_grad[current_ch][w_offset][h_offset] = 0.01;
    }

    return result;
}

//Relu is used for making non-linear transformation of data pass between layers
double leaky_relu(double val) {

    if (val < 0.0) {
        return 0.01 * val;
    }

    return val;
}

//Calculate output for each Convolution layer to use as an input to next layer
double ***conv_forward(conv *layer, double ***input) {

    int out_ch, out_w, out_h;
    double ***output = (double ***)calloc(layer->output_shape[0], sizeof(double **));

    //Matrix Multiplty
    for (out_ch = 0; out_ch < layer->output_shape[0]; out_ch++) {
        output[out_ch] = (double **)calloc(layer->output_shape[1], sizeof(double *));
        for (out_w = 0; out_w < layer->output_shape[1]; out_w++) {
            output[out_ch][out_w] = (double *)calloc(layer->output_shape[2], sizeof(double));
            for (out_h = 0; out_h < layer->output_shape[2]; out_h++) {

                //Calculate convolution
                output[out_ch][out_w][out_h] =
                    convolution(layer, input, out_ch, out_w, out_h);
            }
        }
    }

    return output;
}

//Calculate output after last convolution layer
double *output(conv *prev_layer, softmax *out, double ***input, short backward) {

    int in_ch, in_w, in_h, out_ch;
    double *pass = (double *)calloc(prev_layer->output_shape[0], sizeof(double));
    double *result = (double *)calloc(out->output_dim, sizeof(double));
    double max_coef;

    //Global Average Pooling: Used to reduce the dimension(channel * width * height => channel * 1 * 1)
    for (in_ch = 0; in_ch < prev_layer->output_shape[0]; in_ch++) {
        
        pass[in_ch] = 0.0;

        for (in_w = 0; in_w < prev_layer->output_shape[1]; in_w++) {
            for (in_h = 0; in_h < prev_layer->output_shape[2]; in_h++) {
                pass[in_ch] += input[in_ch][in_w][in_h];
            }
        }

        pass[in_ch] /= (double)(prev_layer->output_shape[1] * prev_layer->output_shape[2]);
    }

    //Softmax Weight Multiply

    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        for (in_ch = 0; in_ch < out->input_dim; in_ch++) {
 
            result[out_ch] += pass[in_ch] * out->weight[in_ch][out_ch];

            if (backward) {
                out->weight_grad[in_ch][out_ch] += pass[in_ch];
            }
        }

        result[out_ch] += out->bias[out_ch];

        if (backward) {
            out->bias_grad[out_ch] += 1.0;
        }
    }

    //Input for Softmax
    max_coef = result[0];
    for (out_ch = 1; out_ch < out->output_dim; out_ch++) {
        if (result[out_ch] > max_coef) max_coef = result[out_ch];
    }

    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        result[out_ch] -= max_coef;
    }

    return result;
}

//Calculate loss between predicted class and true class
double loss(double *result, int label, softmax *out, short backward) {

    //Label smoothed loss
    double exp_sum = 0.0;
    double *probs = (double *)calloc(out->output_dim, sizeof(double));
    int out_ch;

    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        probs[out_ch] = exp(result[out_ch]);
        exp_sum += probs[out_ch];
    }

    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        probs[out_ch] /= exp_sum;
    }

    if (backward) {
        for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
            if (out_ch == label) {
                out->output_grad[out_ch] = probs[out_ch] - 0.9;
            } else {
                out->output_grad[out_ch] = probs[out_ch] - 0.1;
            }
        }
    }

    return -log(probs[label] + 1e-40);
}

//Backpropagation and gradient descent
void update(conv **params, softmax *out, double ****forwards, int depth, double learning_rate, double l2_penalty) {
    int layer, in_ch, in_w, in_h, out_ch, out_w, out_h, fil_w, fil_h, pos_w, pos_h;
    double ***temp_input_grad = NULL;
    double ****temp_filter_grad = NULL;
    double **temp_weight_grad = NULL;
    double *temp_bias_grad = NULL;

    //Applying Chain rule for softmax layer-------------------------------------------------------------------------
    //Making temporal storage
    temp_weight_grad = (double **)calloc(out->input_dim, sizeof(double *));
    for (in_ch = 0; in_ch < out->input_dim; in_ch++) {
        temp_weight_grad[in_ch] = (double *)calloc(out->output_dim, sizeof(double));
    }

    temp_input_grad = (double ***)calloc(params[depth - 1]->output_shape[0], sizeof(double **));
    for (in_ch = 0; in_ch < params[depth - 1]->output_shape[0]; in_ch++) {
        temp_input_grad[in_ch] = (double **)calloc(params[depth - 1]->output_shape[1], sizeof(double *));
        for (in_w = 0; in_w < params[depth - 1]->output_shape[1]; in_w++) {
            temp_input_grad[in_ch][in_w] = (double *)calloc(params[depth - 1]->output_shape[2], sizeof(double));
        }
    }

    //Calculating backpropagation
    for (in_ch = 0; in_ch < out->input_dim; in_ch++) {
        for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
            for (in_w = 0; in_w < params[depth - 1]->output_shape[1]; in_w++) {
                for (in_h = 0; in_h < params[depth - 1]->output_shape[2]; in_h++) {
                    temp_weight_grad[in_ch][out_ch] +=
                        forwards[depth][in_ch][in_w][in_h] *
                        out->output_grad[out_ch];
                    temp_input_grad[in_ch][in_w][in_h] +=
                        out->weight[in_ch][out_ch] *
                        out->output_grad[out_ch];
                    temp_input_grad[in_ch][in_w][in_h] /=
                        (double)(params[depth - 1]->output_shape[1] * params[depth - 1]->output_shape[2]);
                }
            }
        }
    }

    //Replacing output gradient with loss gradient
    for (in_ch = 0; in_ch < out->input_dim; in_ch++) {
        for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
            out->weight_grad[in_ch][out_ch] *= temp_weight_grad[in_ch][out_ch];
        }
        free(temp_weight_grad[in_ch]);
    }
    free(temp_weight_grad);

    for (in_ch = 0; in_ch < params[depth - 1]->output_shape[0]; in_ch++) {
        for (in_w = 0; in_w < params[depth - 1]->output_shape[1]; in_w++) {
            for (in_h = 0; in_h < params[depth - 1]->output_shape[2]; in_h++) {
                params[depth - 1]->output_grad[in_ch][in_w][in_h] *=
                    temp_input_grad[in_ch][in_w][in_h];
            }
            free(temp_input_grad[in_ch][in_w]);
        }
        free(temp_input_grad[in_ch]);
    }

    free(temp_input_grad);
    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        out->bias_grad[out_ch] *= out->output_grad[out_ch];
        out->output_grad[out_ch] = 0.0;
    }

    //Applying Chain rule from output at each convolution layer-----------------------------------------------------------
    for (layer = depth - 1; layer > 0; layer--) {
        //Making temporal storage
        temp_filter_grad = (double ****)calloc(params[layer]->filter_size[0], sizeof(double ***));
        for (in_ch = 0; in_ch < params[layer]->filter_size[0]; in_ch++) {
            temp_filter_grad[in_ch] = (double ***)calloc(params[layer]->filter_size[1], sizeof(double **));
            for (out_ch = 0; out_ch < params[layer]->filter_size[1]; out_ch++) {
                temp_filter_grad[in_ch][out_ch] = (double **)calloc(params[layer]->filter_size[2], sizeof(double *));
                for (fil_w = 0; fil_w < params[layer]->filter_size[2]; fil_w++) {
                    temp_filter_grad[in_ch][out_ch][fil_w] = (double *)calloc(params[layer]->filter_size[3], sizeof(double));
                }
            }
        }

        temp_input_grad = (double ***)calloc(params[layer]->input_shape[0], sizeof(double **));
        for (in_ch = 0; in_ch < params[layer]->input_shape[0]; in_ch++) {
            temp_input_grad[in_ch] = (double **)calloc(params[layer]->input_shape[1], sizeof(double *));
            for (in_w = 0; in_w < params[layer]->input_shape[1]; in_w++) {
                temp_input_grad[in_ch][in_w] = (double *)calloc(params[layer]->input_shape[2], sizeof(double));
            }
        }

        temp_bias_grad = (double *)calloc(params[layer]->output_shape[0], sizeof(double));

        //Calculating per-layer backpropagation
        for (in_ch = 0; in_ch < params[layer]->filter_size[0]; in_ch++) {
            for (out_ch = 0; out_ch < params[layer]->output_shape[0]; out_ch++) {
                for (out_w = 0; out_w < params[layer]->output_shape[1]; out_w++) {
                    for (out_h = 0; out_h < params[layer]->output_shape[2]; out_h++) {
                        for (fil_w = 0; fil_w < params[layer]->filter_size[2]; fil_w++) {
                            for (fil_h = 0; fil_h < params[layer]->filter_size[3]; fil_h++) {
                                pos_w = out_w * (1 + params[layer]->stride_width) + fil_w - (params[layer]->filter_size[2] / 2);
                                pos_h = out_h * (1 + params[layer]->stride_height) + fil_h - (params[layer]->filter_size[3] / 2);
                                if ((0 <= pos_w) && (pos_w < params[layer]->input_shape[1]) &&
                                    (0 <= pos_h) && (pos_h < params[layer]->input_shape[2])) {
                                        temp_input_grad[in_ch][pos_w][pos_h] +=
                                        params[layer]->filter[in_ch][out_ch][fil_w][fil_h] *
                                        params[layer]->output_grad[out_ch][out_w][out_h];
                                    temp_filter_grad[in_ch][out_ch][fil_w][fil_h] +=
                                        forwards[layer][in_ch][pos_w][pos_h] *
                                        params[layer]->output_grad[out_ch][out_w][out_h];
                                    temp_bias_grad[out_ch] +=
                                        params[layer]->output_grad[out_ch][out_w][out_h];
                                }
                            }
                        }
                        params[layer]->output_grad[out_ch][out_w][out_h] = 0.0;
                    }
                }
            }
        }

        //Replacing per-layer output gradients with loss gradients
        for (in_ch = 0; in_ch < params[layer]->filter_size[0]; in_ch++) {
            for (out_ch = 0; out_ch < params[layer]->filter_size[1]; out_ch++) {
                for (fil_w = 0; fil_w < params[layer]->filter_size[2]; fil_w++) {
                    for (fil_h = 0; fil_h < params[layer]->filter_size[3]; fil_h++) {
                        params[layer]->filter_grad[in_ch][out_ch][fil_w][fil_h] =
                            temp_filter_grad[in_ch][out_ch][fil_w][fil_h];
                    }
                    free(temp_filter_grad[in_ch][out_ch][fil_w]);
                }
                free(temp_filter_grad[in_ch][out_ch]);
            }
            free(temp_filter_grad[in_ch]);
        }
        free(temp_filter_grad);

        for (in_ch = 0; in_ch < params[layer]->input_shape[0]; in_ch++) {
            for (in_w = 0; in_w < params[layer]->input_shape[1]; in_w++) {
                for (in_h = 0; in_h < params[layer]->input_shape[2]; in_h++) {
                    params[layer - 1]->output_grad[in_ch][in_w][in_h] *=
                        temp_input_grad[in_ch][in_w][in_h];
                }
                free(temp_input_grad[in_ch][in_w]);
            }
            free(temp_input_grad[in_ch]);
        }
        free(temp_input_grad);

        for (out_ch = 0; out_ch < params[layer]->output_shape[0]; out_ch++) {
            params[layer]->bias_grad[out_ch] *=
                temp_bias_grad[out_ch];
        }
        free(temp_bias_grad);
    }

    //Calculate Backpropagtion of first layer

    temp_filter_grad = (double ****)calloc(params[0]->filter_size[0], sizeof(double ***));
    for (in_ch = 0; in_ch < params[0]->filter_size[0]; in_ch++) {
        temp_filter_grad[in_ch] = (double ***)calloc(params[0]->filter_size[1], sizeof(double **));
        for (out_ch = 0; out_ch < params[0]->filter_size[1]; out_ch++) {
            temp_filter_grad[in_ch][out_ch] = (double **)calloc(params[0]->filter_size[2], sizeof(double *));
            for (fil_w = 0; fil_w < params[0]->filter_size[2]; fil_w++) {
                temp_filter_grad[in_ch][out_ch][fil_w] = (double *)calloc(params[0]->filter_size[3], sizeof(double));
            }
        }
    }

    temp_bias_grad = (double *)calloc(params[0]->output_shape[0], sizeof(double));

    //Calculating per-layer backpropagation
    for (in_ch = 0; in_ch < params[0]->filter_size[0]; in_ch++) {
        for (out_ch = 0; out_ch < params[0]->output_shape[0]; out_ch++) {
            for (out_w = 0; out_w < params[0]->output_shape[1]; out_w++) {
                for (out_h = 0; out_h < params[0]->output_shape[2]; out_h++) {
                    for (fil_w = 0; fil_w < params[0]->filter_size[2]; fil_w++) {
                        for (fil_h = 0; fil_h < params[0]->filter_size[3]; fil_h++) {
                            pos_w = out_w * (1 + params[0]->stride_width) + fil_w - (params[0]->filter_size[2] / 2);
                            pos_h = out_h * (1 + params[0]->stride_height) + fil_h - (params[0]->filter_size[3] / 2);
                            if ((0 <= pos_w) && (pos_w < params[0]->input_shape[1]) &&
                                (0 <= pos_h) && (pos_h < params[0]->input_shape[2])) {
                                temp_filter_grad[in_ch][out_ch][fil_w][fil_h] +=
                                    forwards[0][in_ch][pos_w][pos_h] *
                                    params[0]->output_grad[out_ch][out_w][out_h];
                                temp_bias_grad[out_ch] +=
                                    params[0]->output_grad[out_ch][out_w][out_h];
                            }
                        }
                    }
                    params[0]->output_grad[out_ch][out_w][out_h] = 0.0;
                }
            }
        }
    }

    //Replacing per-layer output gradients with loss gradients at first layer
    for (in_ch = 0; in_ch < params[0]->filter_size[0]; in_ch++) {
        for (out_ch = 0; out_ch < params[0]->filter_size[1]; out_ch++) {
            for (fil_w = 0; fil_w < params[0]->filter_size[2]; fil_w++) {
                for (fil_h = 0; fil_h < params[0]->filter_size[3]; fil_h++) {
                    params[0]->filter_grad[in_ch][out_ch][fil_w][fil_h] =
                        temp_filter_grad[in_ch][out_ch][fil_w][fil_h];
                }
                free(temp_filter_grad[in_ch][out_ch][fil_w]);
            }
            free(temp_filter_grad[in_ch][out_ch]);
        }
        free(temp_filter_grad[in_ch]);
    }
    free(temp_filter_grad);

    for (out_ch = 0; out_ch < params[0]->output_shape[0]; out_ch++)
    {
        params[0]->bias_grad[out_ch] *=
            temp_bias_grad[out_ch];
    }
    free(temp_bias_grad);

    //Update parameters--------------------------------------------------------------------------------------------
    for (layer = 0; layer < depth; layer++) {
        for (in_ch = 0; in_ch < params[layer]->filter_size[0]; in_ch++) {
            for (out_ch = 0; out_ch < params[layer]->filter_size[1]; out_ch++) {
                for (fil_w = 0; fil_w < params[layer]->filter_size[2]; fil_w++){
                    for (fil_h = 0; fil_h < params[layer]->filter_size[3]; fil_h++) {
                        params[layer]->filter[in_ch][out_ch][fil_w][fil_h] *= (1.0 - learning_rate * l2_penalty);
                        params[layer]->filter[in_ch][out_ch][fil_w][fil_h] -=
                            params[layer]->filter_grad[in_ch][out_ch][fil_w][fil_h] * learning_rate;
                        params[layer]->filter_grad[in_ch][out_ch][fil_w][fil_h] = 0.0;
                    }
                }
            }
        }

        for (out_ch = 0; out_ch < params[layer]->output_shape[0]; out_ch++) {
            params[layer]->bias[out_ch] -= params[layer]->bias_grad[out_ch] * learning_rate;
            params[layer]->bias_grad[out_ch] = 0.0;
        }
    }

    for (in_ch = 0; in_ch < out->input_dim; in_ch++) {
        for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
            out->weight[in_ch][out_ch] *= (1.0 - learning_rate * l2_penalty);
            out->weight[in_ch][out_ch] -= out->weight_grad[in_ch][out_ch] * learning_rate;
            out->weight_grad[in_ch][out_ch] = 0.0;
        }
    }

    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        out->bias[out_ch] -= out->bias_grad[out_ch] * learning_rate;
        out->bias_grad[out_ch] = 0.0;
    }
}

//Train the model to make it accurate
double train(conv **params, softmax *out, int depth, img *input, double learning_rate, double l2_penalty) {
    int layer;

    double ****forwards = (double ****)calloc(depth + 1, sizeof(double ***));

    //Calculating per-layer outputs
    forwards[0] = input->data;
    for (layer = 0; layer < depth; layer++) {
        forwards[layer + 1] = conv_forward(params[layer], forwards[layer]);
    }

    //Caculating class probability
    double *softmax_output = output(params[depth - 1], out, forwards[depth], TRUE);

    //Calculating loss
    double result = loss(softmax_output, input->label, out, TRUE);

    //Update the parameters
    update(params, out, forwards, depth, learning_rate, l2_penalty);

    return result;
}

//Input images after training is finished and get result
int inference(conv **params, softmax *out, int depth, img *input, short print) {

    int layer, out_ch;
    double exp_sum = 0.0;
    double ***data = input->data;

    //Passing data each layer
    for (layer = 0; layer < depth; layer++) {
        data = conv_forward(params[layer], data);
    }

    double *result = output(params[layer - 1], out, data, FALSE);
    double *probs = (double *)calloc(out->output_dim, sizeof(double));
    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        probs[out_ch] = exp(result[out_ch]);
        exp_sum += probs[out_ch];
    }

    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        probs[out_ch] /= exp_sum;
    }

    int max_prob_class = 0;

    for (out_ch = 0; out_ch < out->output_dim; out_ch++) {
        if (probs[max_prob_class] < probs[out_ch]) {
            max_prob_class = out_ch;
        }
        if (print) printf("Class %d prob: %.6lf\n", out_ch, probs[out_ch]);
    }

    //Return which class has maximum probabilities
    return max_prob_class;
}

int main() {    
    //Open data files
    FILE *train_set = fopen("../raw/train-images-idx3-ubyte", "r");
    FILE *train_lab = fopen("../raw/train-labels-idx1-ubyte", "r");
    FILE *test_set = fopen("../raw/t10k-images-idx3-ubyte", "r");
    FILE *test_lab = fopen("../raw/t10k-labels-idx1-ubyte", "r");

    //Skip data infomation and go to data start position
    fseek(train_set, 16, SEEK_SET);
    fseek(train_lab, 8, SEEK_SET);
    fseek(test_set, 16, SEEK_SET);
    fseek(test_lab, 8, SEEK_SET);

    //Data information
    //int train_size = 60000, test_size = 10000;
    int width = 28, height = 28;
    int ind = 0;

    //Print Sample Images
    /*
    img *sample_img = load_img(width, height, train_set, train_lab, ind++);
    print_img(sample_img);

    sample_img = load_img(width, height, train_set, train_lab, ind++);
    print_img(sample_img);

    sample_img = load_img(width, height, train_set, train_lab, ind++);
    print_img(sample_img);
    */

    //Neural Network initializing;
    int layer;
    int depth = 3;
    int num_param = 0;
    int *dim_temp = (int *)calloc(3, sizeof(int));

    dim_temp[0] = 1;
    dim_temp[1] = width;
    dim_temp[2] = height;

    conv **params = (conv **)calloc(depth, sizeof(conv *));
    for (layer = 0; layer < depth; layer++) params[layer] = (conv *)calloc(depth, sizeof(conv));

    conv_init(params[0], dim_temp, 16, 3, 3, 1, 1);
    conv_init(params[1], params[0]->output_shape, 32, 3, 3, 1, 1);
    conv_init(params[2], params[1]->output_shape, 64, 3, 3, 1, 1);

    softmax *out = (softmax *)calloc(depth, sizeof(softmax));
    softmax_init(out, params[depth - 1], 10);

    printf("--------------------------------------------------------------------------------\n");
    printf("Model Structure:\n");
    printf("Layer Index\t     Input\t    Output\tWidth\t  Height\n");
        for (int layer = 0; layer < depth; layer++) {
        printf("    Layer%2d\t", layer);
        printf("%3d x%3d x%3d  ", params[layer]->input_shape[0], params[layer]->input_shape[2], params[layer]->input_shape[2]);
        printf("%3d x%3d x%3d  ", params[layer]->output_shape[0], params[layer]->output_shape[1], params[layer]->output_shape[2]);
        printf("%5d\t", params[layer]->filter_size[2]);
        printf("%5d\n", params[layer]->filter_size[3]);
        num_param += (params[layer]->filter_size[0] *
                      params[layer]->filter_size[1] *
                      params[layer]->filter_size[2] *
                      params[layer]->filter_size[3])
                     + params[layer]->output_shape[0];
    }
    printf("    Softmax\t");
    printf("%8d\t", out->input_dim);
    printf("%8d\n", out->output_dim);
    num_param += (out->input_dim + 1) * out->output_dim;
    printf("Total number of parameters: %d\n", num_param);
    printf("--------------------------------------------------------------------------------\n");

    img *data;

    double learning_rate = 6e-6;
    double l2_penalty = 0.1;
    double train_loss;

    for (int j = 0; j < 1; j++) {
        train_loss = 0.0;
        for (ind = 0; ind < 5000; ind++) {
            data = load_img(width, height, train_set, train_lab, ind);
            train_loss += train(params, out, depth, data, learning_rate, l2_penalty);

            //Learning rate scheduling
            learning_rate -= 1e-10;
        }
        printf("epoch%d: %lf\n", j + 1, train_loss / 100.0);
    }

    for (ind = 0; ind < 10; ind++) {
        data = load_img(width, height, test_set, test_lab, ind);
        fseek(test_lab, 8 + ind, SEEK_SET);
        print_img(data);
        printf("%d -> %d\n", data->label, inference(params, out, depth, data, TRUE));
    }

    free(dim_temp);
    for (layer = 0; layer < depth; layer++)
    {
        free(params[layer]);
    }

    free(params);  
    free(out);

    fclose(train_set);
    fclose(train_lab);
    fclose(test_set);
    fclose(test_lab);
    return 0;
}