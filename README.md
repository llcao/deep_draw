# deep_draw
Convolution neural network... for draw video poker. Perhaps, we learn something useful for other poker, too.

# update Dec 14, 2015
Play a game against the computer
> ~/deep_draw/poker-lib/./play_holdem.sh

Train a classifier.
> cd ~/deep_draw/learning/
> python triple_draw_poker_full_output.py

Unfortunately, some useful parameters are in the Python file triple_draw_poker_full_output.py:
—  TRAINING_FORMAT = ‘holdem_events’ # events = bets. This is what you want
—  DATA_FILENAME = '../data/holdem/100k_CNN_holdem_hands.csv’ # location of game CSV data
—  Unzip files like ~/poker-lib/holdem_events_100k_CNN_2_7_vs_other_leaky_ReLU.csv.gz  and move to this location, for more training data
—  MAX_INPUT_SIZE = 740000 # change to much smaller value, to fit in memory
—  VALIDATION_SIZE = 40000 # change this, also
—  NUM_EPOCHS = 20;  NUM_FILTERS = 24;  LEARNING_RATE = 0.02 # can change
— by default, all experimental network shapes, adaptive learning, etc is turned off.

Overview of training setup:
1. __main__ (at the bottom) creates name for model we will pickle at every set. Something like
“ holdem_eventstriple_draw_conv_0.02_learn_rate_20_epoch_adaptive_24_filters_valid_border_1_num_draws_full_hand_hand_context_model.pickle"

2. In def main, we:
A. Load the data. dataset = load_data()
— this involves translating CSV files (one line per poker move) into tensors, and takes forever
(We should batch this in a separate process, if only to have tensors available)
— methods in file draw_poker.py
B. calls build_model() if not using an alternative network layout
C. if (pickled model exists):
     # Load previous weights
     all_param_values_from_file = np.load(out_file)
D. iter_funcs = create_iter_functions_full_output
— this creates loss function, etc

Also note:
— Ignore “adaptive training switch"
— Control-C any time during training to stop training iteration
— model saved (to pickle file) after every iteration of training

———————————

Logically, all of the training functions (including confusing ones related to training with different network shapes) are contained in triple_draw_poker_full_output.py

The functioned used to load data into tensors from CSV, are in draw_poker.py

Many sub-functions in both should be moved to library files, or archived.

———————————

Loss function is computed in create_iter_functions_full_output()

The confusing part is that for “masked objective” (for example bets, where we only know some actions taken), we compute a rather complicated loss function.

output_batch = lasagne.layers.get_output(output_layer)
loss_train_mask = value_action_error(output_batch, z_batch) * m_batch         loss_train_mask = loss_train_mask.mean()

value_action_error()  compares the network output, and training examples

For results from a bet, the comparison is simple. Difference between predicted value, and observed value. For action percentage, it’s a bit more complicated. Most of the logic in value_action_error() has to do with optimizing the value of action percentages.

It would be more correct to compute these loss values separately, and to treat action percentage as a softmax. But we are not doing that, currently.

———————————
 
Settings in ~/.profile (or ~/.bash_profile)

# Settings for GPU
# export THEANO_FLAGS='cuda.root=/usr/local/cuda/,device=gpu,floatX=float32'
# Settings for CPU
export THEANO_FLAGS='blas.ldflags=-lblas -lgfortran'

[or other THEANO_FLAGS]

# Make Deep Draw work
export DRAW_POKER_PYTHON_PATH=$HOME/deep_draw/poker-lib:$HOME/deep_draw/learning:$HOME/deep_draw/poker-lib/CFR
export PYTHONPATH=$DRAW_POKER_PYTHON_PATH:$PYTHONPATH:.

[necessary for Python to know all poker directories]

# Finally, include library path (for installed boost libraries)
export LD_LIBRARY_PATH=/home/ubuntu/deep_draw/poker-lib/CFR:$LD_LIBRARY_PATH

[may or may not be necessary, for boost libraries used to poker CFR library code from C]



