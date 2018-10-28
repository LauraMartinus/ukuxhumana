import tensorflow as tf
import os
from tensor2tensor.utils import trainer_lib
from tensor2tensor import problems
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from translate_entn import TranslateEntnRma
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment, create_hparams


tf.logging.set_verbosity(tf.logging.INFO)
print("Setting up eager execution...")

tf.enable_eager_execution()

# Set a seed so that we have deterministic outputs.
RANDOM_SEED = 301
trainer_lib.set_random_seed(RANDOM_SEED)


print("Setting up the files...")
# Setup and create directories.
DATA_DIR = os.path.expanduser("/tmp/t2t/data")
OUTPUT_DIR = os.path.expanduser("/tmp/t2t/output")
TMP_DIR = os.path.expanduser("/tmp/t2t/tmp")

# Create them.
tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(OUTPUT_DIR)
tf.gfile.MakeDirs(TMP_DIR)

EOS = text_encoder.EOS_ID


print("Generating the data for the translation.....")
entn = TranslateEntnRma()
entn.generate_data(DATA_DIR, TMP_DIR)

print("Viewing the generating data...")
tfe = tf.contrib.eager
Modes = tf.estimator.ModeKeys

# We can iterate over our examples by making an iterator and calling next on it.
eager_iterator = tfe.Iterator(entn.dataset(Modes.EVAL, DATA_DIR))
example = eager_iterator.next()

input_tensor = example["inputs"]
target_tensor = example["targets"]

# The tensors are actually encoded using the generated vocabulary file -- you
# can inspect the actual vocab file in DATA_DIR.
print("Tensor Input: " + str(input_tensor))
print("Tensor Target: " + str(target_tensor))

print("List available problems...")
print(problems.available())
print(registry.list_hparams())


# Setup and create directories.
DATA_DIR = os.path.expanduser("/tmp/t2t/data")
OUTPUT_DIR = os.path.expanduser("/tmp/t2t/output")
TMP_DIR = os.path.expanduser("/tmp/t2t/tmp")

hparams = create_hparams('transformer_base_single_gpu')
hparams.batch_size = 1024
hparams.learning_rate_warmup_steps = 45000
hparams.learning_rate = .4

print("Creating config...")
RUN_CONFIG = create_run_config(
    #model_name='transformer', the pip install does not know what this is
    model_dir=OUTPUT_DIR,
    keep_checkpoint_max=3
)

exp_fn = create_experiment(
    run_config = RUN_CONFIG,
    hparams=hparams,
    model_name='transformer',
    problem_name='translate_entn_rma',
    data_dir=DATA_DIR,
    train_steps=125000,
    eval_steps=100
)

print("Begin training...")
print(exp_fn.train_and_evaluate())
