# Learn-nanoGPT

A simplified, educational version of nanoGPT focused exclusively on Shakespeare text generation with automatic hardware optimization and robust tokenization support.

## About

This is an educational modification of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), streamlined for Shakespeare-only training with enhanced compatibility and user-friendly features.

### Features
- **Automatic GPU/CPU detection** and optimization
- **Shakespeare-focused training** with optimized parameters
- **No configuration files needed** - works out of the box
- **Adaptive model scaling** based on hardware capabilities
- **Multi-format tokenization support** (BPE, character-level, compressed)
- **Robust error handling** with clear debugging information
- **Flash Attention integration** for GPU performance
- **Educational-friendly** with reduced training time (1000 iterations)
- **Kaggle-optimized** for seamless notebook execution

## Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/niloydebbarma-code/Learn-nanoGPT.git
   cd Learn-nanoGPT
   ```

2. **Install Dependencies**
   ```bash
   pip install torch numpy tiktoken requests
   ```

3. **Prepare Data**
   ```bash
   cd data/shakespeare
   python prepare_adaptive.py
   cd ../..
   ```

4. **Start Training**
   ```bash
   python train.py
   ```

5. **Generate Shakespeare Text**
   ```bash
   python sample.py --num_samples=3 --max_new_tokens=200
   ```

## Kaggle Setup (Recommended)

1. **Create New Notebook**
   - Go to Kaggle.com → Create → New Notebook
   - Turn on GPU: Settings → Accelerator → GPU P100

2. **Step-by-Step Cells**
   
   **Cell 1: Setup**
   ```bash
   %%bash
   git clone https://github.com/niloydebbarma-code/Learn-nanoGPT.git
   cd Learn-nanoGPT
   pip install torch numpy tiktoken requests
   ```
   
   **Cell 2: Prepare Data**
   ```bash
   %%bash
   cd Learn-nanoGPT/data/shakespeare
   python prepare_adaptive.py
   ```
   
   **Cell 3: Train Model**
   ```bash
   %%bash
   cd Learn-nanoGPT
   python train.py
   ```
   
   **Cell 4: Generate Text**
   ```bash
   %%bash
   cd Learn-nanoGPT
   python sample.py --num_samples=3 --max_new_tokens=200
   ```

   **Cell 5: Experiment with Parameters**
   ```bash
   %%bash
   cd Learn-nanoGPT
   python sample.py --temperature=0.3 --num_samples=2 --max_new_tokens=150
   ```

**Note**: P100 GPUs provide optimal performance with Flash Attention. Training takes ~15-20 minutes for 1000 iterations.

## Project Structure

```
├── data/shakespeare/
│   ├── prepare_adaptive.py     # Adaptive data preparation (GPU/CPU)
│   ├── prepare.py             # Standard BPE tokenization
│   ├── prepare_gpu.py         # GPU-optimized preparation
│   └── readme.md              # Dataset preparation guide
├── docs/
│   └── Learn-NanoGPT.md       # Detailed documentation
├── train.py                   # Training script
├── model.py                   # GPT transformer model
├── sample.py                  # Text generation script
└── README.md                  # This file
```

## Educational Focus

This project is specifically designed for educational purposes with the following considerations:

- **Reduced training time**: 1000 iterations instead of 5000 for classroom use
- **Clear documentation**: Step-by-step instructions suitable for beginners
- **Kaggle compatibility**: Optimized for popular educational platform
- **Automatic hardware detection**: No manual configuration required
- **Robust error handling**: Clear error messages for troubleshooting
- **Multi-format support**: Works with different tokenization approaches
- **Real-world application**: Demonstrates modern NLP techniques on classic literature



## Troubleshooting

### **Installation Issues**
```bash
pip install torch numpy tiktoken requests --upgrade
```

### **Training Issues**
- **CUDA out of memory**: Reduce batch size or use CPU mode
- **Slow training**: Ensure GPU is enabled in Kaggle settings
- **Loss not decreasing**: Normal for first few iterations, be patient

### **Sampling Issues**
- **"No meta.pkl found"**: Run data preparation first
- **"Meta structure not compatible"**: The script auto-detects format
- **Poor quality text**: Try different temperature values (0.3-1.2)

### **Common Solutions**
- **Directory errors in Kaggle**: Use full path `/kaggle/working/Learn-nanoGPT`
- **Package conflicts**: Restart kernel and reinstall dependencies
- **GPU not detected**: Check Kaggle accelerator settings

### **Expected Results**
- **Training time**: 15-20 minutes on P100 GPU
- **Final loss**: ~1.5-2.0 (validation loss)
- **Model size**: ~30M parameters
- **Generated text**: Coherent Shakespeare-style dialogue with character names

**Training resumes automatically from checkpoints in `out-shakespeare/ckpt.pt`**

## Sample Output

After training, you can expect output like this:

```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief.

KING HENRY:
Once more unto the breach, dear friends, once more;
Or close the wall up with our English dead.
In peace there's nothing so becomes a man
As modest stillness and humility.
```

## Advanced Usage

### **Parameter Experimentation**
```bash
# Conservative generation
python sample.py --temperature=0.3 --num_samples=5

# Creative generation  
python sample.py --temperature=1.2 --num_samples=3

# Longer text
python sample.py --max_new_tokens=500 --num_samples=1

# Custom starting text
python sample.py --start="HAMLET:" --max_new_tokens=300
```

### **Training Customization**
- Modify `max_iters` in `train.py` for longer training
- Adjust `learning_rate` for different convergence speeds
- Change `batch_size` for memory optimization

---

## Contributing

This project is designed for educational use. Feel free to:
- Experiment with different datasets
- Modify model architecture
- Improve documentation
- Add new features

## License

MIT License - See LICENSE file for details.

---

**Happy Learning!**

*"All the world's a stage, and all the men and women merely players."* - William Shakespeare

