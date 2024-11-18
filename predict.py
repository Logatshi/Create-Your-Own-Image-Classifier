import argparse
import json
import numpy as np
import fmodel

def parse_args():
    parser = argparse.ArgumentParser(description='Parser for predict.py')
    parser.add_argument('input', default='./flowers/test/1/image_06743.jpg', nargs='?', type=str)
    parser.add_argument('--dir', default="./flowers/", dest="data_dir")
    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', type=str)
    parser.add_argument('--top_k', default=5, type=int, dest="top_k")
    parser.add_argument('--category_names', default='cat_to_name.json', dest="category_names")
    parser.add_argument('--gpu', default="gpu", dest="gpu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = fmodel.load_checkpoint(args.checkpoint)
    
    with open(args.category_names, 'r') as file:
        category_names = json.load(file)
    
    probabilities = fmodel.predict(args.input, model, args.top_k, args.gpu)
    prob_array = np.array(probabilities[0][0])
    labels = [category_names[str(index + 1)] for index in np.array(probabilities[1][0])]
    
    for label, prob in zip(labels, prob_array):
        print(f"{label} with a probability of {prob:.4f}")
    
    print("finish")

if __name__ == "__main__":
    main()
