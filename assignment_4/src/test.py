import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def test(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total_correct_preds = 0.0
        len_dataset = len(dataloader.dataset)
        targets, outputs = [], []
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            pred = output.argmax(dim=1, keepdim=True)
            correct_preds = pred.eq(y_batch.view_as(pred)).sum().item()
            total_correct_preds += correct_preds
            outputs.extend( pred.view(-1,).detach().cpu().numpy().tolist() )
            targets.extend( y_batch.detach().cpu().numpy().tolist() )

        accuracy = total_correct_preds/float(len_dataset)

    return targets, outputs, accuracy

def get_test_report(target, output, target_names):
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return {label : mat for label, mat in zip(all_cats, confusion_mat)}
