import tarfile
import sys
import os
import glob

if len(sys.argv) < 4:
    print("usage: python compress_datasets_for_t2t.py <root_folder> <language1> <language2>")
root_folder = sys.argv[1]
L1 = sys.argv[2]
L2 = sys.argv[3]

if len(sys.argv) > 4:
    vocab_size = int(sys.argv[4])
else:
    vocab_size = 0

types = ["train", "dev"]

root_folder = os.path.join(root_folder, "%s_%s" % (L1, L2)) 

train_output = '%s_%s.train.tar.gz' % (L1,L2)
dev_output = '%s_%s.dev.tar.gz' % (L1,L2)

train_path = os.path.join(root_folder, train_output)
dev_path = os.path.join(root_folder, dev_output)

# Remove old compressed files
old = glob.glob(os.path.join(root_folder, '*.tar.gz'))
for x in old:
    os.remove(x)

train_files = glob.glob(os.path.join(root_folder, '*.train.*'))
dev_files = glob.glob(os.path.join(root_folder, "*.dev.*"))


with tarfile.open(name=train_path, mode='w:gz') as tar_handle:
    # Find all "trains"
    for t in train_files:
        n = t.split("/")[-1]
        tar_handle.add(t, arcname=n)

    if vocab_size > 0:
        vocab_file = os.path.join(root_folder, "bpe.%d.tokens.vocab" %(vocab_size))
        n = vocab_file.split("/")[-1]
        tar_handle.add(vocab_file, arcname=n)

with tarfile.open(name=dev_path, mode='w:gz') as tar_handle:
    # Find all "trains"
    for t in dev_files:
        n = t.split("/")[-1]
        tar_handle.add(t, arcname=n)

print("Tarballs %s and %s have been created" % (train_path, dev_path))
