import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import settings
import dataset
import metrics
from cnn_mamba import HybridCNNMamba

CHECKPOINT_PATTERN = re.compile(r"(?P<exp>CNNMamba_Res(?P<res>\d+)_Data(?P<data>\d+)_BEST)\.pth$")


def list_checkpoints(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return []

    checkpoints = []
    for filename in sorted(os.listdir(checkpoint_dir)):
        match = CHECKPOINT_PATTERN.match(filename)
        if not match:
            continue

        checkpoints.append(
            {
                "exp_name": match.group("exp"),
                "resolution": int(match.group("res")),
                "data_pct": int(match.group("data")),
                "path": os.path.join(checkpoint_dir, filename),
            }
        )
    return checkpoints


def evaluate_checkpoint(model, loader, criterion):
    model.eval()
    monitor = metrics.MetricMonitor()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Teste", unit="batch", colour="green"):
            imgs = imgs.to(settings.DEVICE)
            labels = labels.to(settings.DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            monitor.update(loss.item(), outputs, labels)

            _, preds = torch.max(outputs, 1)
            preds_all.extend(preds.cpu().numpy().tolist())
            targets_all.extend(labels.cpu().numpy().tolist())

    return monitor.get_results(), targets_all, preds_all


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    output_dir = os.path.join(script_dir, "final_test_results")
    os.makedirs(output_dir, exist_ok=True)

    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"[ERRO] Nenhum checkpoint *_BEST.pth encontrado em: {checkpoint_dir}")
        return

    test_files, test_labels = dataset.get_all_filepaths(settings.TESTE_DIR)
    if len(test_files) == 0:
        print(f"[ERRO] Nenhuma imagem de teste encontrada em: {settings.TESTE_DIR}")
        return

    print("[INFO] Iniciando teste final")
    print(f"[INFO] Device: {settings.DEVICE}")
    print(f"[INFO] Total de checkpoints: {len(checkpoints)}")
    print(f"[INFO] Total de imagens de teste: {len(test_files)}")

    summary_file = os.path.join(output_dir, "FINAL_TEST_SUMMARY.csv")

    for ckpt in checkpoints:
        exp_name = ckpt["exp_name"]
        resolution = ckpt["resolution"]
        ckpt_path = ckpt["path"]

        print(f"\n[INFO] Avaliando {exp_name} (res={resolution})")

        _, test_transform = dataset.get_transforms(resolution)
        test_dataset = dataset.LeukemiaDataset(test_files, test_labels, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=settings.BATCH_SIZE,
            shuffle=False,
            num_workers=settings.NUM_WORKERS,
            pin_memory=settings.PIN_MEMORY,
        )

        model = HybridCNNMamba(num_classes=settings.NUM_CLASSES).to(settings.DEVICE)
        state_dict = torch.load(ckpt_path, map_location=settings.DEVICE)
        model.load_state_dict(state_dict)
        mamba_blocks = model.mamba_depth
        unfrozen_backbone_tensors = model.unfrozen_backbone_param_tensors

        criterion = nn.CrossEntropyLoss()
        monitor_results, targets_test, preds_test = evaluate_checkpoint(model, test_loader, criterion)

        exp_result_id = f"{exp_name}_Mamba{mamba_blocks}_Unfrozen{unfrozen_backbone_tensors}"
        exp_output_dir = os.path.join(output_dir, exp_result_id)
        os.makedirs(exp_output_dir, exist_ok=True)

        overall_file = os.path.join(exp_output_dir, "overall_metrics.csv")
        per_class_file = os.path.join(exp_output_dir, "per_class_metrics.csv")
        cm_csv_file = os.path.join(exp_output_dir, "confusion_matrix.csv")
        cm_png_file = os.path.join(exp_output_dir, "confusion_matrix.png")

        overall = metrics.save_overall_metrics(overall_file, targets_test, preds_test)
        metrics.save_per_class(per_class_file, targets_test, preds_test, settings.CLASSES)
        metrics.save_confusion_matrix_csv(cm_csv_file, targets_test, preds_test, settings.CLASSES)
        metrics.save_confusion_matrix_plot(cm_png_file, targets_test, preds_test, settings.CLASSES)

        summary_row = {
            "result_id": exp_result_id,
            "resolution": resolution,
            "data_pct": ckpt["data_pct"],
            "mamba_blocks": mamba_blocks,
            "unfrozen_backbone_tensors": unfrozen_backbone_tensors,
            "loss": f"{monitor_results['loss']:.6f}",
            "acc": f"{monitor_results['acc']:.6f}",
            "f1": f"{monitor_results['f1']:.6f}",
            "precision": f"{monitor_results['precision']:.6f}",
            "recall": f"{monitor_results['recall']:.6f}",
            "weighted_precision": f"{overall['weighted_precision']:.6f}",
            "weighted_recall": f"{overall['weighted_recall']:.6f}",
            "weighted_f1": f"{overall['weighted_f1']:.6f}",
        }

        metrics.save_results_experiment(summary_file, exp_result_id, summary_row)
        print(
            f"[OK] {exp_result_id} -> Acc: {monitor_results['acc']:.4f} | "
            f"F1: {monitor_results['f1']:.4f} | Macro-F1: {overall['macro_f1']:.4f} | "
            f"Mamba: {mamba_blocks} | Unfrozen tensors: {unfrozen_backbone_tensors}"
        )

    print(f"\n[FINAL] Resultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
