---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:5055
- loss:MultipleNegativesRankingLoss
base_model: snunlp/KR-SBERT-V40K-klueNLI-augSTS
widget:
- source_sentence: 멋사 동아리는 어떤 성격을 갖고 있나요?
  sentences:
  - 기획 파트는 주로 어떤 문제를 다루나요?
  - 멋사는 어떤 사람들이 주로 활동하나요?
  - 프로젝트 안 하고도 활동 가능한가요?
- source_sentence: 협업 잘하는 사람이 많은가요?
  sentences:
  - 회비 없이 참여할 수 있나요?
  - 열정적인 사람이 많은 편인가요?
  - 백엔드 파트 지원서에는 뭘 써야 하나요?
- source_sentence: 멋사에는 팀워크가 좋은 사람들이 많나요?
  sentences:
  - 협업 잘하는 사람이 많은가요?
  - 멋사 지원 일정은 언제 발표되나요?
  - 동아리 모임이 어색하지 않나요?
- source_sentence: 지역 활동도 포함되나요?
  sentences:
  - AI 관련 실습이 포함되어 있나요?
  - 기술 외에도 기획이나 디자인도 하나요?
  - 파트너가 도와주나요?
- source_sentence: 졸업 후에도 계속 참여할 수 있나요?
  sentences:
  - REST API에 대한 이해를 강조하고 싶은데 어떻게 써야 하나요?
  - 졸업 후 어떤 방식으로 참여할 수 있나요?
  - 선발에서 가장 중요한 요소는 무엇인가요?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on snunlp/KR-SBERT-V40K-klueNLI-augSTS
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: sbert eval
      type: sbert-eval
    metrics:
    - type: pearson_cosine
      value: .nan
      name: Pearson Cosine
    - type: spearman_cosine
      value: .nan
      name: Spearman Cosine
---

# SentenceTransformer based on snunlp/KR-SBERT-V40K-klueNLI-augSTS

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [snunlp/KR-SBERT-V40K-klueNLI-augSTS](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [snunlp/KR-SBERT-V40K-klueNLI-augSTS](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS) <!-- at revision 92c6c2c7032f680bff0f9f0c63fadd3f97e635b2 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '졸업 후에도 계속 참여할 수 있나요?',
    '졸업 후 어떤 방식으로 참여할 수 있나요?',
    '선발에서 가장 중요한 요소는 무엇인가요?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `sbert-eval`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| pearson_cosine      | nan     |
| **spearman_cosine** | **nan** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 5,055 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 6 tokens</li><li>mean: 10.27 tokens</li><li>max: 28 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 10.33 tokens</li><li>max: 28 tokens</li></ul> |
* Samples:
  | sentence_0                       | sentence_1                          |
  |:---------------------------------|:------------------------------------|
  | <code>프론트엔드 파트는 주로 뭘 하나요?</code> | <code>프론트엔드는 어떤 기술을 사용하나요?</code>   |
  | <code>주로 어떤 전공의 학생이 많나요?</code>  | <code>멋사에는 팀워크가 좋은 사람들이 많나요?</code> |
  | <code>지원 대상은 어떤가요?</code>        | <code>열정만 있으면 지원할 수 있나요?</code>     |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | sbert-eval_spearman_cosine |
|:------:|:----:|:-------------:|:--------------------------:|
| 1.0    | 316  | -             | nan                        |
| 1.5823 | 500  | 0.6117        | -                          |
| 2.0    | 632  | -             | nan                        |
| 3.0    | 948  | -             | nan                        |


### Framework Versions
- Python: 3.9.21
- Sentence Transformers: 4.1.0
- Transformers: 4.52.3
- PyTorch: 2.6.0+cu118
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->