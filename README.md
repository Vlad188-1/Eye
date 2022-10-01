# Eye

## Чтобы использовать этот репозиторий необходимо установить библотеки:

```bash
pip install -r requirements.txt
```

# Запуск обучения модели

```bash
python train.py --path_to_images path/to/dir/with/images --path_to_masks path/to/dir/with/masks --epochs 100 --encoder "timm-efficientnet-b5" --bacth_size 2 --imgsz 608 --lr 0.003
```

# Запуск инференса

```bash
python train.py --path_to_images path/to/dir/with/images --path_to_weights full/path/to/weights --min_size_area 200 --output_path predicted_masks
```

# Скачать данные и веса можно по сcылкам ниже
1. Данные https://disk.yandex.ru/d/96IPY5S3NSzTww
2. Веса https://disk.yandex.ru/d/NGMi5HO3i_sfBw
