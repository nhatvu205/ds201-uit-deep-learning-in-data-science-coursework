"""
Script hien thi ket qua training
Chay trong Colab cell rieng: %run module_bai_3/display_results.py
"""

def display_training_results():
    from IPython.display import Image, display
    import json
    import os
    
    image_path = 'module_bai_3/training_history.png'
    json_path = 'module_bai_3/training_history.json'
    
    if os.path.exists(image_path):
        print('Training History Plot:')
        display(Image(filename=image_path))
    else:
        print(f'Plot not found at {image_path}')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            history = json.load(f)
        
        print('\nTraining Summary:')
        print(f'  Total epochs: {len(history["train_loss"])}')
        print(f'  Best train F1: {max(history["train_f1"]):.4f}')
        print(f'  Best val F1: {max(history["dev_f1"]):.4f}')
        print(f'  Final train loss: {history["train_loss"][-1]:.4f}')
        print(f'  Final val loss: {history["dev_loss"][-1]:.4f}')
    else:
        print(f'History not found at {json_path}')

if __name__ == '__main__':
    display_training_results()

