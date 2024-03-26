import argparse
import sys
import json
from fastembed import TextEmbedding
from local_fastermbed import get_version


def main():
    parser = argparse.ArgumentParser(
                        prog='List text models',
                        description='List the text models supported by fastembed')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + get_version())
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--table', action='store_true', default=False,
        help = "TSV format")
    group.add_argument('-j', '--json', action='store_true', default=True, 
        help = "JSON format")

    args = parser.parse_args()
    json_format = args.json and not args.table

    models = []
    for model_desc in TextEmbedding.list_supported_models():
        models.append(model_desc)
    if any(models):
        if json_format:
            print(json.dumps(models, indent=4))
        else:
            fields = ["model", "dim", "description", "size_in_GB", "sources.hf", "sources.url"]

            def get_key_path(data: dict, field: str):
                keys = field.split('.')
                depth = len(keys)
                while depth > 0:
                    data = data.get(keys[-depth])
                    if not data:
                        break
                    depth -= 1
                if not data:
                    return ""
                return str(data)

            sep = '\t'
            print(sep.join(fields))
            for model_desc in models:
                row = []
                for field in fields:
                    row.append(get_key_path(model_desc, field))
                print(sep.join(row))
        return 0
    return 1
        
if __name__=="__main__":
    sys.exit(main())
    