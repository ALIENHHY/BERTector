# BERTector
Intrusion detection systems (IDS) are widely used to maintain the stability of network environments, but still face restrictions in generalizability due to the heterogeneity of network traffics. In this work, we propose \textit{BERTector}, a new framework of joint-dataset learning for IDS based on BERT. \textit{BERTector} integrates three key components: NSS-Tokenizer for traffic-aware semantic tokenization, supervised fine-tuning with a hybrid dataset, and low-rank adaptation for efficient fine-tuning. Experiments show that \textit{BERTector} achieves state-of-the-art detection accuracy, strong generalizability, and excellent robustness. \textit{BERTector} achieves the highest accuracy of 99.28\% on NSL-KDD and reaches a 80\% average detection success rate against four mainstream perturbations. These results establishes a unified and efficient solution for modern IDS in complex and dynamic network environments.

File Path:
---

```plaintext
main
├── ablation
│   ├── train-lora.ipynb
│   ├── test-lora.ipynb
│   ├──      ...
├── adversarial
│   ├── Random_Perturbation.ipynb
│   ├── NSL-KDD-10000.csv
├── comparative
│   ├── Deep_Learning.ipynb
│   ├── Machine_Learning.ipynb
├── data
│   ├── sft
│   │   ├── Mixed-sft-500000.csv
│   │   ├── NSL-KDD-100000-sft.csv
│   │   ├── NSL-KDD-400000-sft.csv
│   ├── test
│   │   ├── KDD99-10000.csv
│   │   ├── NSL-KDD-10000.csv
│   │   ├── UNSW_NB15-10000.csv
│   │   ├── X-IIoTID-10000.csv
│   │   ├── NSL-KDD-Gamma.csv
│   │   ├──       ...
├── trainer
│   ├── test_trainer-sft
│   │   ├── checkpoint-1250
│   │   ├──       ...
│   ├── test_trainer-sft-diy
│   │   ├──       ...
│   ├── test_trainer-sft-diy-lora
│   │   ├──       ...
│   ├── test_trainer-sft-diy-lora-mixed
│   │   ├──       ...
│   ├──      ...
├── bert-base-uncased
├── test-sft.ipynb
├── test-sft-diy.ipynb
├── test-sft-diy-lora.ipynb
├── train-sft.ipynb
├── train-sft-diy.ipynb
├── train-sft-diy-lora.ipynb
