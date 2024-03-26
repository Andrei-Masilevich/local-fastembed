import argparse
import sys
import os
from fastembed import TextEmbedding
from fastembed.common.model_management import ModelManagement
from local_fastermbed import get_version, CACHE_DIR


MODEL_ALL="all"

def main():
    parser = argparse.ArgumentParser(
                        prog='Pull text models',
                        description='Pull (download, update) the text models supported by fastembed')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + get_version())
    parser.add_argument('models', metavar='model', type=str, nargs='+',
                        help=f"a list of models (or '{MODEL_ALL}')") 
    parser.add_argument('-s', '--storage', default=CACHE_DIR, 
                        help = f"Local storage (default: {CACHE_DIR})") 

    args = parser.parse_args()

    if not os.path.isdir(args.storage) and not os.path.islink(args.storage):
        raise ValueError(f"Wrong path: {args.storage}")

    supported_models = {}
    for model_desc in TextEmbedding.list_supported_models():
        model = model_desc.get("model")
        supported_models[model] = model_desc
    
    models = set()
    for model in args.models:
        if model == MODEL_ALL:
            models.clear()
            break
        if model not in supported_models.keys():
            raise ValueError(f"{model} - unknown/unsupported!")
        else:
            models.add(model)

    if not any(models):
        models = set(supported_models.keys())

    for model in models:
        print(f"Download {model}")
        ModelManagement.download_model(supported_models.get(model), args.storage, local_files_only = False)
        print()

    return 0
        
if __name__=="__main__":
    sys.exit(main())
