from typing import Callable, List, Tuple
from transformers import AutoModel, AutoTokenizer

# !!! PyTorch requires the large disk space for PC with NVIDEO CUDA support !!!
# }{ to install PyTorch with CPU support only:
#
#    poetry source add --priority=explicit pytorch-cpu-src https://download.pytorch.org/whl/cpu
#    poetry add -G experiments --source pytorch-cpu-src torch
#
#
import torch
import torch.nn.functional as F
from os import path


this_dir_path = path.dirname(path.realpath(__file__))

#sentence-transformers/paraphrase-MiniLM-L6-v2
model_id = path.join(this_dir_path, "models", "paraphrase-MiniLM-L6-v2")
if not path.isdir(model_id):
    raise AssertionError("""
    Downloaded sentence-transformers/paraphrase-MiniLM-L6-v2 model is required.
    
    1. Pull this one to the local folder <your_local>:
           git clone https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
    2. Make symlink to this folder:
           ln -s <your_local> ./models
    """)

class HF:
    """
    HuggingFace Transformer implementation of FlagEmbedding
    """

    def __init__(self, model_id: str):
        self.model = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Way #1
    def embed1(self, texts: List[str]):
        """
        Based on https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        """
        
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings
    
    # Way #2
    def embed2(self, texts: List[str]):
        """
        Based on https://huggingface.co/BAAI/bge-base-en
        """
        
        encoded_input = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        model_output = self.model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = F.normalize(sentence_embeddings)
        return sentence_embeddings
    
hf = HF(model_id=model_id)

documents: List[str] = [
    "Chandrayaan-3 is India's third lunar mission",
    "It aimed to land a rover on the Moon's surface - joining the US, China and Russia",
    "The mission is a follow-up to Chandrayaan-2, which had partial success",
    "Chandrayaan-3 will be launched by the Indian Space Research Organisation (ISRO)",
    "The estimated cost of the mission is around $35 million",
    "It will carry instruments to study the lunar surface and atmosphere",
    "Chandrayaan-3 landed on the Moon's surface on 23rd August 2023",
    "It consists of a lander named Vikram and a rover named Pragyan similar to Chandrayaan-2. Its propulsion module would act like an orbiter.",
    "The propulsion module carries the lander and rover configuration until the spacecraft is in a 100-kilometre (62 mi) lunar orbit",
    "The mission used GSLV Mk III rocket for its launch",
    "Chandrayaan-3 was launched from the Satish Dhawan Space Centre in Sriharikota",
    "Chandrayaan-3 was launched earlier in the year 2023",
]
len(documents)

print(hf.embed1(documents))
print()
print(hf.embed2(documents))
