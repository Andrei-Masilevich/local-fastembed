# ⚡️ FastEmbed extended to support offline mode

About [FastEmbed](https://github.com/qdrant/fastembed/README.md)

---

Given S3-like storage mounted to the folder {/repo}.

In this project FastEmbed hase been extended to use the required sentence transformers only from {/repo} in offline mode.

The following utilities are designed to utilize this new ability:

|Utility|Description|
|-|-|
|list_models|List the text models supported by fastembed|
|pull_models|Pull (download, update) the text models supported by fastembed|
|trace_model|Trace fastembed embedding for the text model for online and offline mode|

### Use case

For instance I intended to use 'sentence-transformers/all-MiniLM-L6-v2' text model in my RAG application.

1. Check if this model is supported:

```bash
./list_models.py -t|cut -f 1|tail -n +2|grep 'sentence-transformers/all-MiniLM-L6-v2'
```

2. In online pull this model to {/repo}:

```bash
./pull_models.py -s {/repo} 'sentence-transformers/all-MiniLM-L6-v2'
```

3. Check in offline mode:

```bash
./trace_model.py --local -s {/repo} -m 'sentence-transformers/all-MiniLM-L6-v2'
```

