docker run --rm --gpus all \
  -v "$PWD:/app" \
  -w /app \
  mi-bci-epoch-window-influence \
  python main.py -m \
  dataset=Dreyer2023 \
  model=Deep4Net \
  epoch_window=w_00_40 \
  'subject=range(1,11)'
