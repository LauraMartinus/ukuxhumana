import tensorflow as tf
import os
from tensor2tensor.utils import trainer_lib
from tensor2tensor import problems
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
import t2t.problems as probs
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment, create_hparams
import sys

print("Setting up eager execution...")
tf.enable_eager_execution()

ukuxhumana_problems = {
    "translate_enaf_rma":probs.translate_enaf.TranslateEnafRma(),
    "translate_ennso_rma":probs.translate_ennso.TranslateEnnsoRma(),
    'translate_entn_rma':probs.translate_entn.TranslateEntnRma(),
    'translate_ents_rma':probs.translate_ents.TranslateEntsRma(),
    'translate_enzu_rma':probs.translate_enzu.TranslateEnzuRma(),
    'translate_enzu_rma_8k':probs.translate_enzu.TranslateEnzuRma8k()
}
t2t-trainer \
  --generate_data \
  --data_dir=/tmp/translate_enzu_rma8k/data \
  --output_dir=/tmp/translate_enzu_rma8k/output \
  --tmp_dir=/tmp/translate_enzu_rma8k/tmp \
  --problem=translate_enzu_rma8k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --train_steps=125000 \
  --eval_steps=100 \
  --t2t_usr_dir=./t2t/problems/

def train(problem_name):
    tf.logging.set_verbosity(tf.logging.INFO)

    problem = ukuxhumana_problems[problem_name]

    # Set a seed so that we have deterministic outputs.
    RANDOM_SEED = 301
    trainer_lib.set_random_seed(RANDOM_SEED)


    print("Setting up the files...")
    # Setup and create directories.
    ROOT_DIR = "/tmp/t2t/%s" % (problem['problem_name'],)
    DATA_DIR = os.path.expanduser("%s/data"  % (ROOT_DIR,))
    OUTPUT_DIR = os.path.expanduser("%s/output" % (ROOT_DIR,))
    TMP_DIR = os.path.expanduser("%s/tmp" % (ROOT_DIR,))

    # Create them.
    tf.gfile.MakeDirs(ROOT_DIR)
    tf.gfile.MakeDirs(DATA_DIR)
    tf.gfile.MakeDirs(OUTPUT_DIR)
    tf.gfile.MakeDirs(TMP_DIR)

    # End-of-sentence character
    EOS = text_encoder.EOS_ID

    print("Generating the data for the translation.....")

    model = problem["model"]
    model.generate_data(DATA_DIR, TMP_DIR)

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
        problem_name=problem_name,
        data_dir=DATA_DIR,
        train_steps=125000,
        eval_steps=100
    )

    print("Begin training...")
    print(exp_fn.train_and_evaluate())
