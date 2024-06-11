What's up, everyone! Today we're diving deep into the world of large language models — we're gonna build GPT-2 (the 124M version) entirely from scratch! Yup, every line of code! We'll start by understanding the GPT-2 architecture, then we'll crank up the speed with some crazy optimizations (using GPUs, mixed precision, the whole shebang!), dial in those hyperparameters, and finally, we'll hit 'run' and come back to see what kind of crazy text our model can generate! Trust me, some of these generations are hilarious. Don't worry if you're new to this, I'll explain everything step-by-step. And if you want to follow along, the code will be on my GitHub (build-nanogpt). This video is basically a live coding session for that repo!

First, we gotta lay the groundwork: We'll explore the GPT-2 architecture, compare it to the original Transformer, and figure out how to implement it in PyTorch. Then, we'll turn our code into a speed demon! We'll leverage GPUs, mixed precision, Torch Compile, and this super cool technique called Flash Attention. Next up, hyperparameters! We'll steal some secrets from the GPT-3 paper and make sure our model trains like a champ. Finally, we'll unleash the beast! We'll train our GPT-2 for a good while, visualize those loss curves, and see if we can beat the performance of the original model. Oh, and we'll get to see some of the crazy, sometimes funny, text it generates along the way!

Now, for those of you who really want to push the limits of speed, I'll also show you llm.c, my C/CUDA implementation of GPT training. It's basically the same thing but way faster because it cuts out all the Python overhead. And of course, all the code for this video will be available on my build-nanogpt repo, complete with detailed commit history so you can see how it all comes together. Let's get coding!

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


## Detailed Walkthrough

### Introduction (00:00:00 - 00:13:47)

Alright everyone, let's get this GPT-2 party started! Our mission today: build the 124 million parameter version from the ground up. Now, when I say "GPT-2," I'm talkin' 'bout a specific model size, not the whole GPT-2 family. There's a bunch of 'em, different sizes, ya know? Like a mini-series! We chart their powers on a graph — bigger model, better performance. We're aiming to beat the OG OpenAI model, which is a fun challenge because they actually released the weights for this one. But their paper is kinda vague on the training deets. So, we're gonna be like code detectives, using the GPT-3 paper for clues on hyperparameters and stuff. Think of it as a remix! GPT-2 architecture with GPT-3 training secrets. To kick things off, we're gonna use Tiny Shakespeare, a super chill dataset, perfect for debugging.

- Explains the goal of reproducing the 124M parameter version of GPT-2.
- Discusses the GPT-2 mini-series and scaling laws in language models.
- Highlights the resources available: OpenAI's blog post, paper, and TensorFlow code, and the Hugging Face Transformers implementation in PyTorch.
- Introduces the Tiny Shakespeare dataset for initial debugging.
- Emphasizes the importance of surpassing the performance of the OpenAI-released model. 

### Exploring the GPT-2 Checkpoint (00:13:47 - 00:13:47)

Before we get our hands dirty building our own GPT-2, let's take a peek under the hood of the pre-trained one from Hugging Face. Kinda like reverse-engineering a sweet piece of tech! We'll load up the model and check out its state dictionary — all the juicy tensors that make it tick. You'll see stuff like token embeddings, positional embeddings, and all those transformer weights.  We'll even visualize those positional embeddings, see if we can spot any cool patterns.  And of course, we gotta make sure this thing actually *works*. We'll sample some text from it, just to confirm we're getting coherent output.

- Loads the pre-trained GPT-2 (124M) model from Hugging Face Transformers.
- Examines the model's state dictionary, highlighting the shapes of key parameters like token embeddings, positional embeddings, and transformer weights.
- Visualizes positional embeddings and discusses their learned structure.
- Samples text from the pre-trained model using Hugging Face pipelines to demonstrate coherent text generation.

### Implementing the GPT-2 nn.Module (00:13:47 - 00:45:50)

Now for the real fun! We're gonna build our own GPT-2 nn.Module from scratch. Think of it like crafting a custom engine for our language machine. Remember that GPT-2 is like a streamlined Transformer, decoder-only, no need for that encoder business. It's also got some layer norm tweaks, like shuffling things around and adding an extra layer norm for good measure. We'll use nn.ModuleDict and nn.ModuleList to keep our code tidy. We'll define our Block module, the heart of the Transformer, with self-attention, layer norms, and an MLP. GELU is our activation function of choice – smoother than ReLU, fewer dead neurons! We'll geek out on multi-headed attention, making it super efficient with PyTorch tensor magic. Then we'll carefully port those Hugging Face weights over to our custom model, gotta make sure everything lines up perfectly. Last but not least, we'll write that forward function, the engine that turns tokens into logits. And to test our creation, we'll generate some text and compare it to the Hugging Face pipeline. Boom!

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

Okay, we've built our GPT-2, but let's face it, training these beasts can be slow. Time is money, people! So we're gonna unleash some serious speed optimization. First, we gotta understand our hardware. GPUs are our friends, especially those A100s with their tensor cores. These bad boys are built for matrix multiplication, which is like 90% of what a Transformer does! But memory bandwidth is the real bottleneck. We gotta move those tensors around fast. That's where reduced precision comes in — TF32, Bfloat16, all these fancy formats that use fewer bits, saving memory and time. Then we have the secret weapon: torch.compile. This compiler wizardry reduces Python overhead and fuses kernels together, making things super efficient. And for attention, we're bringing in the big guns: Flash Attention! It avoids creating that massive attention matrix, saving tons of memory and compute. Finally, remember those "nice" numbers? Changing the vocab size to a power of 2 can magically speed things up because of the way CUDA loves to work. We'll time everything carefully and see those milliseconds melt away. 

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

Time to dial in those hyperparameters! Sadly, the GPT-2 paper is like a cryptic treasure map when it comes to training details. So, we're gonna steal some wisdom from the GPT-3 paper, like those betas and epsilons for our AdamW optimizer. Gradient clipping is our safety net, preventing those wild gradient swings that can mess up training. And for the learning rate, we're going with the classic cosine decay with a warm-up. Start slow, ramp up, then decay gently. It's a proven recipe for success. We'll also use weight decay to keep those parameters in check, kind of like a gentle nudge towards smaller values. It acts as a nice regularizer. Oh, and don't forget about fused AdamW, a sneaky optimization trick that squeezes even more performance out of our optimizer.

- References the GPT-3 paper's appendix for detailed hyperparameter settings.
- Implements AdamW optimizer with specific beta values and epsilon.
- Implements gradient clipping to prevent large gradient updates and destabilizing the optimization.
- Implements a cosine decay learning rate scheduler with warmup and decay to 10% based on token counts, following the GPT-3 schedule.
- Discusses the complexity of hyperparameter relationships and the decision to primarily follow OpenAI's settings.
- Implements weight decay with a value of 0.1, applying it only to specific parameter groups like embeddings and weights involved in matrix multiplications.
- Discusses the benefits of weight decay as a regularization technique.
- Introduces fused AdamW, a more efficient implementation of AdamW, and achieves a small speedup by using it.

### Gradient Accumulation & Distributed Data Parallel (02:46:52 - 03:10:21)

Alright, now we're getting into the pro optimization strategies. Sometimes you wanna use a big batch size, but your GPU is like, "Nope, not enough memory!" That's where gradient accumulation comes in clutch. It's like batch size hacking! We do multiple forward-backward passes, accumulating those gradients, and then update the parameters. Just gotta remember to scale the loss correctly, otherwise our optimization gets wonky. Now, if you're lucky enough to have multiple GPUs, we're gonna unleash the power of Distributed Data Parallel (DDP). Imagine a team of GPUs working together, each one training on a different part of the data. We'll use torchrun to launch those parallel processes and make sure they're all in sync. It's like a GPU symphony!

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

Let's talk datasets! Data is the fuel that powers these language models. GPT-2 was trained on this mysterious "WebText" dataset, scraped from Reddit links, but sadly, they never released it. GPT-3 went all out, using Common Crawl, WebText2, books, Wikipedia — a massive data buffet! But Common Crawl, let me tell ya, it's a wild jungle of text. Lots of noise and junk in there. Needs serious filtering. There are some curated datasets out there like Red Pajama, but my personal favorite is FineWebEDU. It's like Common Crawl, but cleaned up and focused on educational content. And you know what? A 10 billion token sample is enough to get us close to GPT-level performance.

- Discusses the WebText dataset used in GPT-2, which was scraped from Reddit outbound links and never publicly released.
- Mentions OpenWebText as an attempt to reproduce the WebText dataset.
- Examines the GPT-3 training dataset, which includes Common Crawl, WebText2, books, and Wikipedia.
- Highlights the challenges of using Common Crawl due to its noisy nature and the need for careful filtering.
- Introduces the Red Pajama dataset and its slimmed-down version as good examples of curated data mixtures.
- Focuses on the FineWeb dataset and its educational subset (FineWebEDU), which offers high-quality filtered Common Crawl data.
- Chooses to use a 10 billion token sample of FineWebEDU for training, considering it sufficient for achieving competitive performance.

### Validation Data Split & Evaluation (03:23:10 - 03:28:23)

We gotta keep our model honest, so we'll split our dataset into training and validation sets.  Validation loss is our reality check, telling us if we're overfitting.  We'll peek at it during training, just to make sure things are going smoothly.  But it's not a perfect comparison to the pre-trained GPT-2 because that model was trained on a different data diet.  For a more objective measure of how awesome our model is, we'll use HellaSwag. It's a tricky sentence completion task, like a multiple-choice quiz for language models.  We'll adapt it to a token completion format, so even our smaller models can handle it.  Lower loss means better completion, and we'll aim to beat the scores of those pre-trained GPT-2 champs.

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

Time for the big reveal! We'll check out those loss curves after our overnight training session.  Did we beat the GPT-2 baseline?  You bet! But there might be some weird bumps in the loss, maybe due to the way FineWebEDU is structured.  We could try shuffling the data shards around to smooth things out.  The exciting part: our model almost matches the HellaSwag accuracy of GPT-3, even though it trained on way fewer tokens. Data quality matters, folks! Now, for the grand finale, let's generate some text from our model.  After all this hard work, we gotta see what kind of language magic it can do.  

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

Whew! That was a whirlwind tour of building GPT-2 from scratch. We learned about the architecture, optimized for speed like crazy, figured out those hyperparameters, and even got our model spitting out some decent text.  Not bad for a day's work!  Remember, this is just the pre-training step. If we wanna create a chatty GPT, we'd need to fine-tune it on conversational data. But that's a story for another day.  I'll be posting all the code on GitHub, so you can play with it, experiment, and maybe even teach me a thing or two.  Feel free to hit me up on Discord if you have any questions or wanna geek out about GPT-2. Until next time, keep coding, and let those language models sing!

- Recapitulates the entire process of reproducing GPT-2 (124M) from scratch, highlighting achievements and remaining challenges.
- Points to the `build-nanogpt` repository on GitHub for the complete code and commit history.
- Encourages viewers to engage in discussions, raise issues, or contribute pull requests on GitHub and participate in the Zero to Hero Discord server for further interaction. 
