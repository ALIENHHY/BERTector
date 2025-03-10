# NIDS-with-LLM
NIDS with LLM and LORA

# File Path:
main

├── data

│   ├── sft

│   │   ├── Mixed-sft-500000.csv

│   │   ├── NSL-KDD-100000-sft.csv

│   ├── test

│   │   ├── CICIDS2018-10000.csv

│   │   ├── KDD99-10000.csv

│   │   ├── NSL-KDD-10000.csv

│   │   ├── UNSW_NB15-10000.csv

│   │   ├── X-IIoTID-10000.csv

├── trainer

│   ├── test_trainer-sft

│   │   ├── checkpoint-1250

│   │   ├── checkpoint-2500

                  ...

│   ├── test_trainer-sft-diy

                  ...

│   ├── test_trainer-sft-diy-lora

                  ...

│   ├── test_trainer-sft-diy-lora-mixed

                  ...

├── bert-base-uncased

├── test-sft.ipynb

├── test-sft-diy.ipynb

├── test-sft-diy-lora.ipynb

├── train-sft.ipynb

├── train-sft-diy.ipynb

├── train-sft-diy-lora.ipynb
