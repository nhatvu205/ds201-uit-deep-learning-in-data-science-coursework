import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import matplotlib.pyplot as plt

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def translate_batch(model, src, config, tgt_vocab):
    model.eval()
    with torch.no_grad():
        translations = model.translate(
            src, 
            config.max_length, 
            tgt_vocab.word2idx[tgt_vocab.SOS_TOKEN],
            tgt_vocab.word2idx[tgt_vocab.EOS_TOKEN]
        )
    return translations

def evaluate_model(model, dataloader, config, tgt_vocab):
    download_nltk_data()
    
    model.eval()
    
    references_all = []
    hypotheses_all = []
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = {'rougeL': []}
    
    print("Generating translations...")
    for batch in tqdm(dataloader):
        src = batch['src'].to(config.device)
        tgt_texts = batch['tgt_text']
        
        translations = translate_batch(model, src, config, tgt_vocab)
        
        for i in range(len(tgt_texts)):
            reference = tgt_texts[i].split()
            hypothesis = tgt_vocab.decode(translations[i].cpu().numpy()).split()
            
            references_all.append([reference])
            hypotheses_all.append(hypothesis)
            
            ref_str = ' '.join(reference)
            hyp_str = ' '.join(hypothesis)
            
            rouge_result = rouge_scorer_obj.score(ref_str, hyp_str)
            for key in rouge_scores:
                rouge_scores[key].append(rouge_result[key].fmeasure)
        
    
    smoothing = SmoothingFunction().method1
    
    rougeL_avg = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
    
    results = {
        'ROUGE-L': rougeL_avg
    }
    
    print("\nEvaluation Results:")
    print("="*50)
    for metric, score in results.items():
        print(f"{metric:12s}: {score:.4f}")
    print("="*50)
    
    return results

def show_translation_examples(model, dataloader, config, src_vocab, tgt_vocab, num_examples=5):
    model.eval()
    
    print("\nTranslation Examples:")
    print("="*80)
    
    batch = next(iter(dataloader))
    src = batch['src'][:num_examples].to(config.device)
    src_texts = batch['src_text'][:num_examples]
    tgt_texts = batch['tgt_text'][:num_examples]
    
    translations = translate_batch(model, src, config, tgt_vocab)
    
    for i in range(num_examples):
        translation_text = tgt_vocab.decode(translations[i].cpu().numpy())
        translation_indices = translations[i].cpu().numpy().tolist()
        
        print(f"\nExample {i+1}:")
        print(f"Source:      {src_texts[i]}")
        print(f"Reference:   {tgt_texts[i]}")
        print(f"Translation: {translation_text}")
        print(f"Indices:     {translation_indices[:10]}...") 
        print("-"*80)

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss) + 1
    plt.axvline(x=min_val_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch: {min_val_epoch}')
    plt.plot(min_val_epoch, min_val_loss, 'g*', markersize=15, label=f'Best Loss: {min_val_loss:.4f}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    print("\nTraining History Summary:")
    print("="*50)
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {min_val_loss:.4f} (Epoch {min_val_epoch})")
    print("="*50)

