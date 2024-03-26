import argparse
import sys
import os
import numpy as np
from typing import List
from fastembed import TextEmbedding
from local_fastermbed import get_version, CACHE_DIR


DEFAULT_MODEL_ID='sentence-transformers/all-MiniLM-L6-v2'

def main():
    parser = argparse.ArgumentParser(
                        prog='Trace text model',
                        description='Trace fastembed embedding for the text model for online and offline mode')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + get_version())
    parser.add_argument('-s', '--storage', default=CACHE_DIR, 
                        help = f"Local storage (default: {CACHE_DIR})") 
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL_ID, 
                        help = f"Id (default: {DEFAULT_MODEL_ID})")
    parser.add_argument('-l', '--local', action='store_true', default=False,
                        help = "offline only")

    args = parser.parse_args()

    if not os.path.isdir(args.storage) and not os.path.islink(args.storage):
        raise ValueError(f"Wrong path: {args.storage}")

    documents: List[str] = [
        "He fought against the Mughal Empire led by Akbar.",
        "Как правило это многофункциональная система построена по модульному принципу.",
        "Su vida ha sido retratada en varias películas, programas de televisión y libros.",
    ]
    
    loop = 0
    if args.local:
        loop = 1
    while loop < 2:
        loop += 1
        print(f"For local_files_only = {loop==2}")
        embedding_model = TextEmbedding(model_name=args.model, 
                                        max_length=512, 
                                        local_files_only=loop==2,
                                        cache_dir=args.storage)
        
        embeddings_generator = embedding_model.embed(documents, batch_size=10)
        embeddings_list = list(embeddings_generator)
        
        print(f"Done for {len(embeddings_list)} documents.")
        print()
    return 0

if __name__=="__main__":
    sys.exit(main())