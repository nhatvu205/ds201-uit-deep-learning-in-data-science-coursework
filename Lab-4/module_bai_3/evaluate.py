import torch
from tqdm import tqdm
from rouge_score import rouge_scorer

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
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    print("Generating translations for ROUGE-L evaluation...")
    for batch in tqdm(dataloader):
        src = batch['src'].to(config.device)
        tgt_texts = batch['tgt_text']
        translations = translate_batch(model, src, config, tgt_vocab)
        for i in range(len(tgt_texts)):
            ref = tgt_texts[i]
            hyp = tgt_vocab.decode(translations[i].cpu().numpy())
            score = scorer.score(ref, hyp)['rougeL'].fmeasure
            rouge_l_scores.append(score)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    print(f'ROUGE-L: {avg_rouge_l:.4f}')
    return avg_rouge_l

def show_translation_examples(model, dataloader, config, src_vocab, tgt_vocab, num_examples=5):
    model.eval()
    batch = next(iter(dataloader))
    src = batch['src'][:num_examples].to(config.device)
    src_texts = batch['src_text'][:num_examples]
    tgt_texts = batch['tgt_text'][:num_examples]
    translations = translate_batch(model, src, config, tgt_vocab)
    for i in range(num_examples):
        translation_text = tgt_vocab.decode(translations[i].cpu().numpy())
        print(f'\nExample {i+1}')
        print(f'Source:      {src_texts[i]}')
        print(f'Reference:   {tgt_texts[i]}')
        print(f'Translation: {translation_text}')

