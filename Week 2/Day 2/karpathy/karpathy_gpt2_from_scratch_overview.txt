We reproduce the GPT-2 (124M) from scratch. This video covers the whole process: First we build the GPT-2 network, then we optimize its training to be really fast, then we set up the training run following the GPT-2 and GPT-3 paper and their hyperparameters, then we hit run, and come back the next morning to see our results, and enjoy some amusing model generations. Keep in mind that in some places this video builds on the knowledge from earlier videos in the Zero to Hero Playlist (see my channel). You could also see this video as building my nanoGPT repo, which by the end is about 90% similar.


## Executive Summary: Reproducing GPT-2 (124M)

This video provides a comprehensive walkthrough of reproducing the GPT-2 (124M) language model from scratch using PyTorch. The process involves:

1. **Understanding the GPT-2 Architecture:** Examining the original GPT-2 code, exploring the Hugging Face Transformers implementation, and comparing it to the original transformer architecture from the "Attention is All You Need" paper.
2. **Implementing the GPT-2 Network:** Building a PyTorch `nn.Module` for GPT-2 from scratch, including token and positional embeddings, transformer blocks, MLPs, and the attention mechanism.
3. **Optimizing for Speed:** Leveraging GPUs, mixed precision (TF32 and Bfloat16), Torch Compile, and Flash Attention to drastically improve training speed and throughput.
4. **Implementing Hyperparameters and Optimization:** Applying AdamW with appropriate betas, gradient clipping, a cosine decay learning rate scheduler with warmup, weight decay, and fused AdamW based on the GPT-3 paper.
5. **Gradient Accumulation and Distributed Data Parallel:** Using gradient accumulation to simulate large batch sizes on smaller GPUs and employing Distributed Data Parallel (DDP) to train across multiple GPUs.
6. **Dataset Selection and Processing:** Examining datasets used in GPT-2 and GPT-3, selecting the FineWebEDU dataset for training, and implementing a data loader to handle sharded data.
7. **Validation and Evaluation:** Implementing validation loss calculation on a held-out split of the dataset and incorporating the HellaSwag evaluation benchmark to assess model performance.
8. **Training and Results:** Training the model for one and four epochs, visualizing the loss curves, achieving performance comparable to or exceeding the OpenAI GPT-2 (124M) checkpoint, and generating text samples to observe model improvement.

This video also briefly introduces `llm.c`, a highly optimized C/CUDA implementation of GPT training that achieves faster performance compared to the PyTorch code while maintaining identical computational results. The accompanying code repository (`build-nanogpt`) provides a step-by-step implementation of the entire process with detailed git commit history.

## Detailed Walkthrough

### Introduction (00:00:00 - 00:13:47)

- Explains the goal of reproducing the 124M parameter version of GPT-2.
- Discusses the GPT-2 mini-series and scaling laws in language models.
- Highlights the resources available: OpenAI's blog post, paper, and TensorFlow code, and the Hugging Face Transformers implementation in PyTorch.
- Introduces the Tiny Shakespeare dataset for initial debugging.
- Emphasizes the importance of surpassing the performance of the OpenAI-released model.

### Exploring the GPT-2 Checkpoint (00:13:47 - 00:13:47)

- Loads the pre-trained GPT-2 (124M) model from Hugging Face Transformers.
- Examines the model's state dictionary, highlighting the shapes of key parameters like token embeddings, positional embeddings, and transformer weights.
- Visualizes positional embeddings and discusses their learned structure.
- Samples text from the pre-trained model using Hugging Face pipelines to demonstrate coherent text generation.

### Implementing the GPT-2 nn.Module (00:13:47 - 00:45:50)

- Compares the GPT-2 architecture to the original transformer model, noting the absence of the encoder and cross-attention in GPT-2.
- Highlights modifications made in GPT-2: layer norm reshuffling and an additional layer norm before the final classifier.
- Implements the skeleton structure of the GPT-2 `nn.Module` using `nn.ModuleDict` and `nn.ModuleList` to organize submodules.
- Defines the `Block` module, which comprises self-attention, layer norms, and an MLP, following the GPT-2 paper's modifications.
- Implements the MLP using two linear layers with GELU (Gaussian Error Linear Units) activation and discusses the benefits of GELU over ReLU.
- Explains the multi-headed attention mechanism and provides an efficient PyTorch implementation using transposes and reshaping.
- Loads weights from the Hugging Face model into the custom GPT-2 class, ensuring proper alignment of tensor shapes and naming conventions.
- Implements the `forward` function to process input tokens and generate logits for the next token prediction.
- Generates text from the custom model initialized with GPT-2 weights and compares the results with Hugging Face pipeline outputs.

### Let's Make It Fast (01:22:18 - 02:14:55)

- Analyzes available GPU resources and performance capabilities, specifically focusing on the A100 GPU.
- Introduces the concept of tensor cores and their role in accelerating matrix multiplications.
- Discusses the importance of memory bandwidth and the benefits of reduced precision in deep learning training.
- Implements and analyzes performance with TF32 (TensorFloat32) precision, achieving a 3x speedup compared to FP32.
- Implements and analyzes performance with Bfloat16 (Brain Floating Point) precision, achieving a slight speedup over TF32 and highlighting its advantages over FP16.
- Introduces Torch Compile, explaining its role in reducing Python overhead and enabling kernel fusion for further optimization.
- Implements Torch Compile and achieves a 2.3x speedup over Bfloat16, showcasing its effectiveness in accelerating training.
- Implements Flash Attention, a kernel fusion algorithm that optimizes attention computation by avoiding the materialization of the large attention matrix.
- Achieves a 27% speedup with Flash Attention over Torch Compile, demonstrating its effectiveness for specialized operations beyond Torch Compile's capabilities.
- Optimizes vocab size to a power of 2 (50304) to improve kernel efficiency and achieves a 4% speedup, emphasizing the importance of "nice" numbers in CUDA computations.

### Hyperparameters, AdamW, Gradient Clipping (02:14:55 - 02:46:52)

- References the GPT-3 paper's appendix for detailed hyperparameter settings.
- Implements AdamW optimizer with specific beta values and epsilon.
- Implements gradient clipping to prevent large gradient updates and destabilizing the optimization.
- Implements a cosine decay learning rate scheduler with warmup and decay to 10% based on token counts, following the GPT-3 schedule.
- Discusses the complexity of hyperparameter relationships and the decision to primarily follow OpenAI's settings.
- Implements weight decay with a value of 0.1, applying it only to specific parameter groups like embeddings and weights involved in matrix multiplications.
- Discusses the benefits of weight decay as a regularization technique.
- Introduces fused AdamW, a more efficient implementation of AdamW, and achieves a small speedup by using it.

### Gradient Accumulation & Distributed Data Parallel (02:46:52 - 03:10:21)

- Explains the need for gradient accumulation to simulate large batch sizes when GPU memory is limited.
- Calculates the required gradient accumulation steps based on the desired total batch size and micro batch size.
- Modifies the training loop to accumulate gradients for multiple forward-backward passes before updating parameters.
- Highlights a critical detail in gradient accumulation: the need to scale the loss by the number of accumulation steps to maintain a correct loss objective.
- Demonstrates the correct implementation of gradient accumulation and confirms that it produces equivalent optimization results compared to using a single large batch.
- Introduces Distributed Data Parallel (DDP) for training across multiple GPUs.
- Explains the setup process for DDP using `torch.run` and the environment variables it sets: rank, local rank, and world size.
- Modifies the code to handle multiple processes, assigning GPUs based on local rank and ensuring that each process has its own data loader.
- Wraps the model in the DDP container and explains the synchronization of gradients using `allReduce` after the backward pass.
- Disables gradient synchronization during gradient accumulation to avoid unnecessary communication between processes.
- Averages the accumulated loss across all processes using `dist.allReduce` to obtain a global loss value.

### Datasets Used in GPT-2 & GPT-3 (03:10:21 - 03:23:10)

- Discusses the WebText dataset used in GPT-2, which was scraped from Reddit outbound links and never publicly released.
- Mentions OpenWebText as an attempt to reproduce the WebText dataset.
- Examines the GPT-3 training dataset, which includes Common Crawl, WebText2, books, and Wikipedia.
- Highlights the challenges of using Common Crawl due to its noisy nature and the need for careful filtering.
- Introduces the Red Pajama dataset and its slimmed-down version as good examples of curated data mixtures.
- Focuses on the FineWeb dataset and its educational subset (FineWebEDU), which offers high-quality filtered Common Crawl data.
- Chooses to use a 10 billion token sample of FineWebEDU for training, considering it sufficient for achieving competitive performance.

### Validation Data Split & Evaluation (03:23:10 - 03:28:23)

- Modifies the data loader to handle the FineWebEDU shards and introduce a validation split.
- Adds code to periodically evaluate the validation loss and generate text samples during training.
- Discusses the importance of validation loss for monitoring overfitting, especially for multi-epoch training.
- Explains the limitations of comparing validation loss to the pre-trained GPT-2 model due to differences in training data distribution.
- Introduces the HellaSwag evaluation benchmark as a more objective and standard measure of model performance.
- Explains the concept of HellaSwag as a sentence completion task with adversarially chosen incorrect options to challenge language models.
- Discusses the need for adapting HellaSwag to a token completion format for evaluating smaller language models that cannot handle multiple choice directly.
- Presents the method of evaluating HellaSwag by calculating the average cross-entropy loss for each option and choosing the one with the lowest loss.
- Shows the HellaSwag accuracy scores for the pre-trained GPT-2 models as targets to surpass during training.

### Results in the Morning (03:43:05 - 03:59:39)

- Visualizes the training and validation loss curves after running the training overnight for four epochs (40 billion tokens).
- Observes that the model surpasses the validation loss of the pre-trained GPT-2 (124M) model but exhibits periodic fluctuations in loss potentially related to the FineWebEDU dataset's structure.
- Notes that the model nearly matches the HellaSwag accuracy of the GPT-3 (124M) model, despite training on only 40 billion tokens compared to GPT-3's 300 billion.
- Discusses potential explanations for this efficiency gain, including differences in training data distribution, dataset quality, and potential overlap between HellaSwag and FineWebEDU.
- Highlights further improvements to the data loader, such as permuting data shards and documents to mitigate the observed loss periodicity.
- Generates text samples from the overnight trained model and observes significant improvements in coherence and self-awareness compared to the model trained for only one epoch.
- Discusses the possibility of further fine-tuning the pre-trained model for conversational tasks like chatbots using supervised fine-tuning (SFT).
- Briefly introduces `llm.c`, a C/CUDA implementation of GPT training that achieves faster performance than the PyTorch code while maintaining identical computational results.
- Compares the performance and output of the PyTorch implementation and `llm.c`, demonstrating the speed advantage of the C/CUDA implementation.

### Summary and Resources (03:59:39 - 03:59:39)

- Recapitulates the entire process of reproducing GPT-2 (124M) from scratch, highlighting achievements and remaining challenges.
- Points to the `build-nanogpt` repository on GitHub for the complete code and commit history.
- Encourages viewers to engage in discussions, raise issues, or contribute pull requests on GitHub and participate in the Zero to Hero Discord server for further interaction.