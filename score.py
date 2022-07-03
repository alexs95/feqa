import pandas as pd

from loader import load
from tqdm import tqdm
from pathlib import Path
from feqa import FEQA
import argparse
import os

# python -m spacy download en_core_web_sm
# python -m spacy en
import benepar
import nltk
benepar.download('benepar_en2')
nltk.download('stopwords')


def compute_score(storyids, claimids, sentids, scores):
    df = pd.DataFrame({
        "storyid": storyids,
        "claimid": claimids,
        "sentid": sentids,
        "score": scores
    })
    return df.groupby(['storyid', 'claimid']).max()['score'].mean()


def preprocess(ids, stories, summaries):
    stries = []
    smmries = []
    storyids = []
    claimids = []
    sentids = []
    for storyid, story, summary in tqdm(zip(ids, stories, summaries), "Creating examples", total=len(stories)):
        for claimid, claim in enumerate(summary):
            print(claim)
            text_a = " ".join(s.strip() for s in story)
            text_b = claim.strip()
            stries.append(text_a)
            smmries.append(text_b)
            storyids.append(storyid)
            claimids.append(claimid)
            sentids.append(0)

    return storyids, claimids, sentids, stries, smmries


def parse_args():
    base = Path(__file__).parent.parent.resolve()
    evaluation = os.path.join(base, "evaluation", "1000_sample")
    data = os.path.join(base, "data/cnndm")
    pointer_gen_cov = os.path.join(base, "/pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--paragraph', action='store_true', default=False)
    parser.add_argument('--coref', action='store_true', default=False)
    parser.add_argument('--cnndm', type=str, help='Path to cnn/dm dataset.', default=data)
    parser.add_argument('--summaries', type=str, help='Path to decoded summaries.', default=pointer_gen_cov)
    parser.add_argument('--evaluation', type=str, help='Path to evaluation directory.', default=evaluation)
    parser.add_argument('--samples', type=int, help='Number of stories to preprocess.', default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    feqaScorer = FEQA(use_gpu=args.gpu)
    ids, stories, summaries = load(args.cnndm, args.summaries, args.coref, args.samples)
    print(ids)
    storyids, claimids, sentids, stories, summaries = preprocess(ids, stories, summaries)
    scores = feqaScorer.compute_score(stories, summaries, aggregate=False)
    score = compute_score(storyids, claimids, sentids, scores)
    print(args.evaluation)
    print(score)