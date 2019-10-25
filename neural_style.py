import os

#https://stackoverflow.com/a/53014306
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# https://stackoverflow.com/a/42121886
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from argparse import ArgumentParser
import imageio
import tensorflow as tf

from utils import load_vgg_model, generate_noise_image, save_image 
from utils import reshape_and_normalize_style, reshape_and_normalize_content




# default arguments
CONTENT = 'images/sample_1.jpg'
CONTENT_WEIGHT = 1e1
STYLE = 'images/van_gogh.jpg'
STYLE_WEIGHT = 4e2
STYLE_LAYER_WEIGHT_FACTOR = 1
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

LEARNING_RATE = 1e1
ITERATIONS = 500
PRINT_ITERATIONS = 10
CHECKPOINT_ITERATIONS = 10
HEIGHT = 300
WIDTH = 400
OUTPUT_FOLDER = 'output/'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', 
                    dest='content', help='content image',
                    metavar='CONTENT', required=True)
    parser.add_argument('--style', 
                    dest='style', help='style image',
                    metavar='STYLE', required=True)
    parser.add_argument('--output-folder', 
                    dest='output_folder', help='output folder path',
                    metavar='OUTPUT_FOLDER', default=OUTPUT_FOLDER)
    parser.add_argument('--vgg', 
                    dest='vgg_path', help='vgg path(default %(default)s)',
                    metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--width', type=int,
                    dest='width', help='output width',
                    metavar='WIDTH')
    parser.add_argument('--height', type=int,
                    dest='height', help='output height',
                    metavar='HEIGHT')
    
    parser.add_argument('--print-iterations', type=int,
                    dest='print_iterations', help='print frequency for model statistics(default %(default)s)',
                    metavar='PRINT_ITERATIONS', default = PRINT_ITERATIONS)
    parser.add_argument('--checkpoint-iterations', type=int,
                    dest='checkpoint_iterations', help='frequency to save generated image(default %(default)s)',
                    metavar='CHECKPOINT_ITERATIONS', default = CHECKPOINT_ITERATIONS)
    parser.add_argument('--iterations', type=int,
                    dest='iterations', help='iterations(default %(default)s)',
                    metavar='ITERATIONS', default=ITERATIONS)
    
    parser.add_argument('--learning-rate', type=float,
                    dest='learning_rate', help='learning rate (default %(default)s)',
                    metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--content-weight', type=float,
                    dest='content_weight', help='content weight (default %(default)s)',
                    metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                    dest='style_weight', help='style weight (default %(default)s)',
                    metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    
    parser.add_argument('--style-layer-weight-factor', type=float,
                    dest='style_layer_weight_factor',
                    help='style layer weight increase as - '
                    'weight(layer<n+1>) = weight_factor*weight(layer<n>)(default %(default)s)',
                    metavar='STYLE_LAYER_WEIGHT_FACTOR', default=STYLE_LAYER_WEIGHT_FACTOR)
    
    return parser


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Parameters
    ------------
    a_C : tensor of dimension (1, n_H, n_W, n_C), hidden layer activations of content image 
    a_G : tensor of dimension (1, n_H, n_W, n_C), hidden layer activations of generated image
    
    Returns
    ---------
    J_content : scalar that you compute using equation 1 above.
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C = tf.reshape(a_C, shape=[-1,n_C])
    a_G = tf.reshape(a_G, shape=[-1,n_C])

    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C,a_G)))
    
    return J_content


def gram_matrix(A):
    """
    Parameters
    ------------
    A : matrix of shape (n_C, n_H*n_W)
    
    Returns
    -------------
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A,tf.transpose(A))    
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Computes style cost considering single layer activations.
    
    Parameters
    ------------
    a_S : tensor of dimension (1, n_H, n_W, n_C), hidden layer activations of style image
    a_G : tensor of dimension (1, n_H, n_W, n_C), hidden layer activations of generated image
    
    Returns
    --------
    J_style_layer : tensor representing a scalar value, style cost 
    """
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(tf.transpose(a_S),shape=[n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G),shape=[n_C, n_H*n_W])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = (1/(4*n_H*n_W*n_C*n_H*n_W*n_C))*(tf.reduce_sum(tf.square(tf.subtract(GS,GG))))
     
    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Parameters
    ------------
    model : our tensorflow model
    STYLE_LAYERS : A python list of tuple (_,_) containing:
                        - the names of the layers we would like to extract style from, and
                        - a coefficient for each of them
    
    Returns
    --------
    J_style : tensor representing a scalar value, style cost
    """
    
    J_style = 0

    # Loop over all layers to calculate total cost
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function (style + content)
    
    Parameters
    ------------
    J_content : content cost coded above
    J_style : style cost coded above
    alpha : hyperparameter weighting the importance of the content cost
    beta : hyperparameter weighting the importance of the style cost
    
    Returns
    --------
    J : tensor representing a scalar value, total cost 
    """
    
    J = tf.add(tf.multiply(J_content,alpha),tf.multiply(J_style, beta))  
    return J



if __name__ == "__main__":
    
    parser = build_parser()
    options = parser.parse_args()
    
    if not os.path.isfile(options.vgg_path):
        parser.error("Unable to locate VGG parameters at %s" % options.vgg_path)
    
    # Read and modify content and style image
    content_image = imageio.imread(options.content)
    
    if options.width is None or options.height is None:      
        HEIGHT = content_image.shape[0]
        WIDTH = content_image.shape[1]
    else:
        HEIGHT = options.height
        WIDTH = options.width
        
    content_image = reshape_and_normalize_content(content_image, HEIGHT, WIDTH)

    style_image = imageio.imread(options.style)
    style_image = reshape_and_normalize_style(style_image, HEIGHT, WIDTH)
    
    # Generate noise image to use as input in style transfer network
    generated_image = generate_noise_image(content_image[0])
    
    # Computing style layer weights for different conv layers
    factor = options.style_layer_weight_factor
    style_weights = [1]
    for i in range(4):
        style_weights.append(style_weights[i]*factor)
    STYLE_LAYERS = [
        ('conv1_1', style_weights[0]*1.0/sum(style_weights)),
        ('conv2_1', style_weights[1]*1.0/sum(style_weights)),
        ('conv3_1', style_weights[2]*1.0/sum(style_weights)),
        ('conv4_1', style_weights[3]*1.0/sum(style_weights)),
        ('conv5_1', style_weights[4]*1.0/sum(style_weights))]
      
    
    # Start tensorflow interactive session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    model = load_vgg_model(options.vgg_path,HEIGHT,WIDTH)
    
    # Assign the content image to be the input of the VGG model and calculates content cost  
    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)  
    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out
    J_content = compute_content_cost(a_C, a_G)
    
    
    # Assign the input of the model to be the "style" image  and calculates style cost
    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(model, STYLE_LAYERS)
    J = total_cost(J_content, J_style, alpha = options.content_weight, beta = options.style_weight)
    
    # Running tf Model
    optimizer = tf.train.AdamOptimizer(options.learning_rate)
    train_step = optimizer.minimize(J)
    num_iterations = options.iterations
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(generated_image))
    
    for i in range(num_iterations):  
        sess.run(train_step)
        generated_image = sess.run(model['input'])
    
        if i%(options.print_iterations) == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))      
            
        if i%(options.checkpoint_iterations) == 0:
            save_image(options.output_folder + str(i) + ".png", generated_image[0])
    
    # save last generated image  
    generated_image = sess.run(model['input'])
    save_image(options.output_folder+"generated.png", generated_image[0])
    
