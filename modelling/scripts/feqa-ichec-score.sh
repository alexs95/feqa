#!/bin/sh
#SBATCH -p GpuQ
#SBATCH --nodes 1
#SBATCH --time 04:00:00
#SBATCH -A ngcom023c
#SBATCH --mail-user=a.shapovalov1@nuigalway.ie
#SBATCH --mail-type=ALL

module load cuda/11.2
module load conda/2

conda init
conda activate summarization3.7

cd $SLURM_SUBMIT_DIR
python3 score.py --evaluation test_decoded_resolved --coref --cnndm $PWD/../data/cnndm --summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded --gpu
# python3 score.py --evaluation test_decoded_unresolved --cnndm $PWD/../data/cnndm --summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded --gpu
# python3 score.py --evaluation test_reference_resolved --coref --cnndm $PWD/../data/cnndm --summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference --gpu
# python3 score.py --evaluation test_reference_unresolved --cnndm $PWD/../data/cnndm --summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference --gpu
# python3 score.py --evaluation val_decoded_resolved --coref --cnndm $PWD/../data/cnndm --summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded --gpu
# python3 score.py --evaluation val_decoded_unresolved --cnndm $PWD/../data/cnndm --summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded --gpu
# python3 score.py --evaluation val_reference_resolved --coref --cnndm $PWD/../data/cnndm --summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference --gpu
# python3 score.py --evaluation val_reference_unresolved --cnndm $PWD/../data/cnndm --summaries $PWD/../pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference --gpu