import tensorflow as tf
import os
from tensor2tensor.utils import trainer_lib
from tensor2tensor import problems
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
import problems as probs
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment, create_hparams
import sys

def get_our_problems(vocab_size):
    our_problems = {
        "translate_enaf_rma":{
            'source': 'en',
            'target': 'af',
            'problem_name': 'translate_enaf_rma',
            'prefix': 'en_af_%s' % vocab_size,
            'model': probs.translate_enaf.TranslateEnafRma(vocab_size),
        },
        "translate_ennso_rma":{
            'source': 'en',
            'target': 'nso',
            'problem_name': 'translate_ennso_rma',
            'prefix': 'en_nso_%s' % vocab_size,
            'model': probs.translate_ennso.TranslateEnnsoRma(vocab_size),
        },
        'translate_entn_rma':{
            'source': 'en',
            'target': 'tn',
            'problem_name': 'translate_entn_rma',
            'prefix': 'en_tn_%s' % vocab_size,
            'model': probs.translate_entn.TranslateEntnRma(vocab_size),
        },
        'translate_ents_rma':{
            'source': 'en',
            'target': 'ts',
            'problem_name': 'translate_ents_rma',
            'prefix': 'en_ts_%s' % vocab_size,
            'model': probs.translate_ents.TranslateEntsRma(vocab_size),
        },
        'translate_enzu_rma':{
            'source': 'en',
            'target': 'zu',
            'problem_name': 'translate_enzu_rma',
            'prefix': 'en_zu_%s' % vocab_size,
            'model': probs.translate_enzu.TranslateEnzuRma(vocab_size),
        }
    }
    return our_problems


def train(problem):
    problem_name = problem['problem_name']
    for k in problem:
        print("%s: %s" % (k,problem[k]))

    print("Setting up the files...")
    # Setup and create directories.
    ROOT_DIR = "/tmp/t2t/%s" % (problem['prefix'],)
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

    print("Viewing the generating data...")
    tfe = tf.contrib.eager
    Modes = tf.estimator.ModeKeys

    # We can iterate over our examples by making an iterator and calling next on it.
    eager_iterator = tfe.Iterator(model.dataset(Modes.EVAL, DATA_DIR))
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
